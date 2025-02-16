import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, infostate_tensor_shape, num_actions, hidden_dim, device, dtype=torch.float32):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.register_buffer('num_actions', torch.tensor(num_actions, dtype=torch.int32))
        self.value_fc0 = nn.Linear(infostate_tensor_shape, hidden_dim, device=device, dtype=dtype)
        self.value_fc1 = nn.Linear(hidden_dim, 1, device=device, dtype=dtype)
        self.policy_fc0 = nn.Linear(infostate_tensor_shape, hidden_dim, device=device, dtype=dtype)
        self.policy_fc1 = nn.Linear(hidden_dim, num_actions, device=device, dtype=dtype)

    def forward(self, information_state, legal_actions):
        logit_list, log_policy_list, policy_list, value_list = [], [], [], []
        for t in range(0, len(information_state)):
            observations = information_state[t]
            filter_row = legal_actions[t]
            value = self.value_fc1(torch.relu(self.value_fc0(observations)))
            logits = self.policy_fc1(torch.relu(self.policy_fc0(observations)))
            exp_logits = torch.where(filter_row, torch.exp(logits), 0)
            policy = torch.nn.functional.normalize(exp_logits, dim=-1, p=1.)
            log_sum = torch.log(torch.sum(exp_logits, dim=-1, keepdim=True))
            log_policy = torch.where(filter_row, logits - log_sum, 0)
            logit_list.append(logits)
            log_policy_list.append(log_policy)
            policy_list.append(policy)
            value_list.append(value)

        return (
            torch.stack(logit_list, dim=0),
            torch.stack(log_policy_list, dim=0),
            torch.stack(policy_list, dim=0),
            torch.stack(value_list, dim=0)
        )
