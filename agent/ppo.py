import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from .memory import PPOBuffer
from .model import Actor, Critic


class PPO:
    def __init__(self, num_users, num_states, num_points, num_parts, num_embeds, num_involve, lr_a, lr_c, gamma, lam,
                 repeat_time, batch_size, eps_clip, w_entropy):
        self.actor = Actor(num_states, num_points=num_points, num_parts=num_parts, num_embeds=num_embeds,
                           num_involve=num_involve).cuda()
        self.critic = Critic(num_states).cuda()

        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), lr_a)
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr_c)

        self.buffer = PPOBuffer()

        self.gamma = gamma
        self.lam = lam
        self.repeat_time = repeat_time
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.w_entropy = w_entropy

    def select_action(self, state, test=False):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).cuda()

            prob_points, prob_parts, prob_embeds, prob_involve = self.actor(state)

            point_list = [3, 5, 8]
            parts_list = [1, 2, 4, 8]
            embed_list = [1024, 2048, 4096]

            # point
            dist_point = Categorical(prob_points)
            point = dist_point.sample()

            # part
            dist_part = Categorical(prob_parts)
            part = dist_part.sample()
            # power
            dist_embed = Categorical(prob_embeds)
            embed = dist_embed.sample()
            # involve
            dist_involve = Categorical(prob_involve)
            involve = dist_involve.sample()

            if not test:
                self.buffer.actor_buffer.actions_point.append(point)
                self.buffer.actor_buffer.actions_part.append(part)
                self.buffer.actor_buffer.actions_embed.append(embed)
                self.buffer.actor_buffer.actions_involve.append(involve)

                self.buffer.actor_buffer.log_probs_point.append(dist_point.log_prob(point))
                self.buffer.actor_buffer.log_probs_part.append(dist_part.log_prob(part))
                self.buffer.actor_buffer.log_probs_embed.append(dist_embed.log_prob(embed))
                self.buffer.actor_buffer.log_probs_involve.append(dist_involve.log_prob(involve))

            self.buffer.actor_buffer.info['prob_points'] = prob_points.tolist()
            self.buffer.actor_buffer.info['prob_parts'] = prob_parts.tolist()
            self.buffer.actor_buffer.info['prob_embeds'] = prob_embeds.tolist()
            self.buffer.actor_buffer.info['prob_involve'] = prob_involve.tolist()

        return point_list[point.item()], parts_list[part.item()], embed_list[embed.item()], involve.item()

    def update(self):
        self.buffer.build_tensor()
        # for generate gae
        with torch.no_grad():
            pred_values_buffer = self.critic(self.buffer.states_tensor).squeeze()

        target_values_buffer, advantages_buffer = self.get_gae(pred_values_buffer)
        for _ in range(int(self.repeat_time * (len(self.buffer) / self.batch_size))):
            indices = torch.randint(len(self.buffer), size=(self.batch_size,), requires_grad=False).cuda()
            state = self.buffer.states_tensor[indices]
            target_values = target_values_buffer[indices]
            advantages = advantages_buffer[indices]

            loss_point = []
            loss_part = []
            loss_embed = []
            loss_involve = []

            logprobs_point = self.buffer.actor_buffer.log_probs_point_tensor[indices]
            logprobs_part = self.buffer.actor_buffer.log_probs_part_tensor[indices]
            logprobs_embed = self.buffer.actor_buffer.log_probs_embed_tensor[indices]
            logprobs_involve = self.buffer.actor_buffer.log_probs_involve_tensor[indices]

            new_logprobs_point, new_logprobs_part, new_logprobs_embed, new_logprobs_involve, \
                entropy_point, entropy_part, entropy_embed, entropy_involve = self.eval(state, indices)
            self.buffer.actor_buffer.info['entropy_point'] = entropy_point.mean().item()
            self.buffer.actor_buffer.info['entropy_part'] = entropy_part.mean().item()
            self.buffer.actor_buffer.info['entropy_embed'] = entropy_embed.mean().item()
            self.buffer.actor_buffer.info['entropy_involve'] = entropy_involve.mean().item()

            # point
            ratio_point = (new_logprobs_point - logprobs_point).exp()
            surr1_point = advantages * ratio_point
            surr2_point = advantages * torch.clamp(ratio_point, 1 - self.eps_clip, 1 + self.eps_clip)
            loss_point.append((-torch.min(surr1_point, surr2_point) - self.w_entropy * entropy_point).mean())
            # part
            ratio_part = (new_logprobs_part - logprobs_part).exp()
            surr1_part = advantages * ratio_part
            surr2_part = advantages * torch.clamp(ratio_part, 1 - self.eps_clip, 1 + self.eps_clip)
            loss_part.append((-torch.min(surr1_part, surr2_part) - self.w_entropy * entropy_part).mean())
            # embed
            ratio_embed = (new_logprobs_embed - logprobs_embed).exp()
            surr1_embed = advantages * ratio_embed
            surr2_embed = advantages * torch.clamp(ratio_embed, 1 - self.eps_clip, 1 + self.eps_clip)
            loss_embed.append((-torch.min(surr1_embed, surr2_embed) - self.w_entropy * entropy_embed).mean())
            # involve
            ratio_involve = (new_logprobs_involve - logprobs_involve).exp()
            surr1_involve = advantages * ratio_involve
            surr2_involve = advantages * torch.clamp(ratio_involve, 1 - self.eps_clip, 1 + self.eps_clip)
            loss_involve.append((-torch.min(surr1_involve, surr2_involve) - self.w_entropy * entropy_involve).mean())

            loss_a = torch.stack(loss_point + loss_part + loss_embed + loss_involve).mean()

            self.optimizer_a.zero_grad()
            loss_a.backward()
            self.optimizer_a.step()

            pred_values = self.critic(state).squeeze()
            self.optimizer_c.zero_grad()
            loss_c = F.mse_loss(pred_values, target_values)
            # loss_c = F.smooth_l1_loss(pred_values, target_values)
            loss_c.backward()
            self.optimizer_c.step()
            self.buffer.info['loss_value'] = loss_c.item()

        self.buffer.init()

        return loss_a.cpu().detach().numpy(), loss_c.cpu().detach().numpy()

    def eval(self, state, indices):
        actions_point = self.buffer.actor_buffer.actions_point_tensor[indices]
        actions_part = self.buffer.actor_buffer.actions_part_tensor[indices]
        actions_embed = self.buffer.actor_buffer.actions_embed_tensor[indices]
        actions_involve = self.buffer.actor_buffer.actions_involve_tensor[indices]

        prob_points, prob_parts, prob_embeds, prob_involve = self.actor(state)

        dist_point = Categorical(prob_points)
        dist_part = Categorical(prob_parts)
        dist_embed = Categorical(prob_embeds)
        dist_involve = Categorical(prob_involve)

        logprobs_point = dist_point.log_prob(actions_point)
        logprobs_part = dist_part.log_prob(actions_part)
        logprobs_embed = dist_embed.log_prob(actions_embed)
        logprobs_involve = dist_involve.log_prob(actions_involve)

        entropy_point = dist_point.entropy()
        entropy_part = dist_part.entropy()
        entropy_embed = dist_embed.entropy()
        entropy_involve = dist_involve.entropy()

        return (logprobs_point, logprobs_part, logprobs_embed, logprobs_involve,
                entropy_point, entropy_part, entropy_embed, entropy_involve)

    def get_gae(self, pred_values_buffer):
        with torch.no_grad():
            target_values_buffer = torch.empty(len(self.buffer), dtype=torch.float32).cuda()
            advantages_buffer = torch.empty(len(self.buffer), dtype=torch.float32).cuda()

            next_value = 0
            next_advantage = 0
            for i in reversed(range(len(self.buffer))):
                reward = self.buffer.rewards_tensor[i]
                mask = 1 - self.buffer.is_terminals_tensor[i]
                # value
                target_values_buffer[i] = reward + mask * self.gamma * next_value
                # GAE
                delta = reward + mask * self.gamma * next_value - pred_values_buffer[i]
                advantages_buffer[i] = delta + mask * self.lam * next_advantage

                next_value = pred_values_buffer[i]
                next_advantage = advantages_buffer[i]
            advantages_buffer = (advantages_buffer - advantages_buffer.mean()) / (advantages_buffer.std() + 1e-5)
        return target_values_buffer, advantages_buffer

    def save_model(self, filename, args, info=None):
        dic = {'actor': self.actor.state_dict(),
               'critic': self.critic.state_dict(),
               'args': args,
               'info': info}
        torch.save(dic, filename)

    def load_model(self, actor_dict, critic_dict):
        self.actor.load_state_dict(actor_dict)
        self.critic.load_state_dict(critic_dict)
