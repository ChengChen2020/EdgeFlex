import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, num_states=11, num_points=3, num_parts=4, num_embeds=3):
        super(Actor, self).__init__()
        self.base = nn.Sequential(nn.Linear(num_states, 256),
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

    def forward(self, x):
        code = self.base(x)
        prob_points = self.point_header(code)
        prob_parts = self.channel_header(code)
        prob_embeds = self.embed_header
        # power_mu = self.power_mu_header(code) * (self.pmax - 1e-10) + 1e-10
        # power_sigma = self.power_sigma_header(code)
        return prob_points, prob_parts, prob_embeds


class Critic(nn.Module):
    def __init__(self, num_states=11):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(num_states, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)
