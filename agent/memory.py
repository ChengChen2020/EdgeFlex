import torch


class ActorBuffer:
    def __init__(self):
        self.actions_point = []
        self.log_probs_point = []
        self.actions_part = []
        self.log_probs_part = []
        self.actions_embed = []
        self.log_probs_embed = []
        self.actions_involve = []
        self.log_probs_involve = []
        self.info = {}

    def build_tensor(self):
        self.actions_point_tensor = torch.tensor(self.actions_point, dtype=torch.float32).cuda()
        self.log_probs_point_tensor = torch.tensor(self.log_probs_point, dtype=torch.float32).cuda()
        self.actions_part_tensor = torch.tensor(self.actions_part, dtype=torch.float32).cuda()
        self.log_probs_part_tensor = torch.tensor(self.log_probs_part, dtype=torch.float32).cuda()
        self.actions_embed_tensor = torch.tensor(self.actions_embed, dtype=torch.float32).cuda()
        self.log_probs_embed_tensor = torch.tensor(self.log_probs_embed, dtype=torch.float32).cuda()
        self.actions_involve_tensor = torch.tensor(self.actions_involve, dtype=torch.float32).cuda()
        self.log_probs_involve_tensor = torch.tensor(self.log_probs_involve, dtype=torch.float32).cuda()

    def init(self):
        self.actions_point.clear()
        self.log_probs_point.clear()
        self.actions_part.clear()
        self.log_probs_part.clear()
        self.actions_embed.clear()
        self.log_probs_embed.clear()
        self.actions_involve.clear()
        self.log_probs_involve.clear()
        del self.actions_point_tensor
        del self.log_probs_point_tensor
        del self.actions_part_tensor
        del self.log_probs_part_tensor
        del self.actions_embed_tensor
        del self.log_probs_embed_tensor
        del self.actions_involve_tensor
        del self.log_probs_involve_tensor


class PPOBuffer:
    def __init__(self):
        self.states = []
        self.rewards = []
        self.is_terminals = []
        self.actor_buffer = ActorBuffer()
        self.info = {}

    def build_tensor(self):
        self.states_tensor = torch.tensor(self.states, dtype=torch.float32).cuda()
        self.rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32).cuda()
        self.is_terminals_tensor = torch.tensor(self.is_terminals, dtype=torch.float32).cuda()
        self.actor_buffer.build_tensor()

    def init(self):
        self.states.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        del self.states_tensor
        del self.rewards_tensor
        del self.is_terminals_tensor
        self.actor_buffer.init()

    def __len__(self):
        return len(self.states)
