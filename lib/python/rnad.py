import pyspiel
import torch
from torch import nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(
            self, embedding_dim, dropout=0.0, prenorm=True, activation=nn.ReLU()
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            activation,
            nn.Dropout(dropout),
        )
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.prenorm = prenorm
        nn.init.trunc_normal_(self.layer[0].weight, std=0.02, a=-0.04, b=0.04)
        self.de

    def forward(self, x):
        if self.prenorm:
            return x + self.layer(self.layernorm(x))

        return self.layernorm(x + self.layer(x))


class RNadModel(nn.Module):
    def __init__(self, infostate_tensor_shape, num_actions, hidden_dim, dropout):
        super().__init__()

        self.tower = nn.Sequential(
            nn.Linear(infostate_tensor_shape, hidden_dim),
            ResNet(hidden_dim, dropout),
            ResNet(hidden_dim, dropout),
            ResNet(hidden_dim, dropout),
        )

        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, information_state, legal_actions):
        embedding = self.tower(information_state)

        logits = self.policy_head(embedding)
        exp_logits = torch.where(legal_actions, torch.exp(logits), 0)
        policy = torch.nn.functional.normalize(exp_logits, dim=-1, p=1)
        log_sum = torch.log(torch.sum(exp_logits, dim=-1, keepdim=True))
        log_policy = torch.where(legal_actions, logits - log_sum, 0)

        value = self.value_head(embedding)

        return (logits, log_policy, policy, value)


class RNaD(nn.Module):
    def __init__(self, game_def):
        self.game = pyspiel.load_game(game_def)
        self.model = self.init_model()

        self.target_model = self.init_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.reg_model = self.init_model()
        self.reg_model.load_state_dict(self.model.state_dict())
        self.reg_model_prev = self.init_model()
        self.reg_model_prev.load_state_dict(self.model.state_dict())

    def init_model(self):
        infostate_tensor_shape = self.game.information_state_tensor_shape()[0]
        num_actions = self.game.num_distinct_actions()
        model = RNadModel(
            infostate_tensor_shape=infostate_tensor_shape,
            num_actions=num_actions,
            hidden_dim=256,
            dropout=0.1
        )
        return model

    def forward(self, trajectories):
        information_state = torch.stack([
            torch.tensor(t.information_state)
            for t in trajectories
        ])
        legal_actions = torch.stack([
            torch.tensor(t.legal_actions)
            for t in trajectories
        ])

        logit, log_pi, pi, v = self.model(information_state, legal_actions)