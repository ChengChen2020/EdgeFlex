import torch
from torch import nn
from torch.distributions import MultivariateNormal

import time
import numpy as np


class Storage:
    def __init__(self):
        self.actions = []
        self.values = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.returns = []

    def compute_returns(self, next_value, gamma):
        # compute returns for advantages
        returns = np.zeros(len(self.rewards) + 1)
        returns[-1] = next_value
        for i in reversed(range(len(self.rewards))):
            returns[i] = returns[i + 1] * gamma * (1 - self.is_terminals[i]) + self.rewards[i]
            self.returns.append(torch.tensor([returns[i]]))
        self.returns.reverse()

    def clear_storage(self):
        self.actions.clear()
        self.values.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.returns.clear()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, num_points=3, num_parts=5, num_embeds=3, exploration_param=0.05, device="cpu"):
        super(ActorCritic, self).__init__()
        # output of actor in [0, 1]
        self.base = nn.Sequential(nn.Linear(state_dim, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 128),
                                  nn.ReLU())
        self.point_header = nn.Sequential(nn.Linear(128, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, num_points),
                                          nn.Softmax(dim=-1))
        self.part_header = nn.Sequential(nn.Linear(128, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, num_parts),
                                         nn.Softmax(dim=-1))
        self.embed_header = nn.Sequential(nn.Linear(128, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, num_embeds),
                                          nn.Softmax(dim=-1))
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # self.device = device
        # self.action_var = torch.full((action_dim,), exploration_param ** 2).to(self.device)
        # self.random_action = True

    def forward(self, state):
        value = self.critic(state)
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        if not self.random_action:
            action = action_mean
        else:
            action = dist.sample()

        action_logprobs = dist.log_prob(action)

        return action.detach(), action_logprobs, value

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value = self.critic(state)

        return action_logprobs, torch.squeeze(value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, exploration_param, lr, betas, gamma, ppo_epoch, ppo_clip, use_gae=False):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.ppo_clip = ppo_clip
        self.ppo_epoch = ppo_epoch
        self.use_gae = use_gae

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim, exploration_param, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, exploration_param, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state, storage):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, action_logprobs, value = self.policy_old.forward(state)

        storage.logprobs.append(action_logprobs)
        storage.values.append(value)
        storage.states.append(state)
        storage.actions.append(action)

        point = [3, 5, 8]
        parts = [1, 2, 4, 8]
        embed = [1024, 2048, 4096]

        action = action.squeeze().detach().numpy()
        ida = int(action[0] * 3)
        ida = 2 if ida == 3 else ida
        idb = int(action[1] * 4)
        idb = 3 if idb == 4 else idb
        idc = int(action[2] * 3)
        idc = 2 if idc == 3 else idc
        idd = int(action[3] * 16)
        idd = 15 if idd == 16 else idd
        return point[ida], parts[idb], embed[idc], idd

    def get_value(self, state):
        return self.policy_old.critic(state)

    def update(self, storage, state):
        episode_policy_loss = 0
        episode_value_loss = 0
        if self.use_gae:
            raise NotImplementedError
        advantages = (torch.tensor(storage.returns) - torch.tensor(storage.values)).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(storage.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(storage.actions).to(self.device), 1).detach()
        old_action_logprobs = torch.squeeze(torch.stack(storage.logprobs), 1).to(self.device).detach()
        old_returns = torch.squeeze(torch.stack(storage.returns), 1).to(self.device).detach()

        for t in range(self.ppo_epoch):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_action_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (state_values - old_returns).pow(2).mean()
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            episode_policy_loss += policy_loss.detach()
            episode_value_loss += value_loss.detach()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return episode_policy_loss / self.ppo_epoch, episode_value_loss / self.ppo_epoch

    def save_model(self, data_path):
        torch.save(self.policy.state_dict(),
                   '{}ppo_{}.pth'.format(data_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))


if __name__ == '__main__':
    policy = ActorCritic(11, 3)
    state = torch.randn(11)
    print(policy.forward(state))
