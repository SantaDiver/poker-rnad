import pyspiel
from torch import nn


class ResNet(nn.Module):
    def __init__(
            self, embedding_dim, dropout=0.0, prenorm=True, activation=nn.ReLU()):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            activation,
            nn.Dropout(dropout),
        )
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.prenorm = prenorm
        nn.init.trunc_normal_(self.layer[0].weight, std=0.02, a=-0.04, b=0.04)

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
        self.policy_tower = nn.Linear(hidden_dim, num_actions)

        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_policy_head = nn.Sequential(
            self.policy_tower,
            nn.LogSoftmax(dim=-1)
        )
        self.policy_head = nn.Sequential(
            self.policy_tower,
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        embedding = self.tower(x)
        return (
            self.value_head(embedding),
            self.log_policy_head(embedding),
            self.policy_head(embedding)
        )


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
        pass