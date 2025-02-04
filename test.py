import pyspiel
import torch
from torch import nn

import time
from poker_rnad_py import Actor


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


class RNad:
    def __init__(self, game_def):
        self.game = pyspiel.load_game(game_def)
        infostate_tensor_shape = self.game.information_state_tensor_shape()[0]
        num_actions = self.game.num_distinct_actions()

        self.device = torch.device("cpu")
        self.model = RNadModel(
            infostate_tensor_shape=infostate_tensor_shape,
            num_actions=num_actions,
            hidden_dim=256,
            dropout=0.1
        ).to(self.device)
        jit_model = torch.jit.script(self.model)
        self.actor = Actor(
            game=self.game,
            model=jit_model._c,
            num_workers=2,
            num_worked_threads=0,
            batch_size=1024,
            max_queue_capacity=16,
            device_name=str(self.device)
        )

    def step(self):
        self.actor.run()

        for _ in range(3):
            print(len(self.actor.get_batch(wait_seconds=1)))

        jit_model = torch.jit.script(self.model)
        self.actor.update_model(model=jit_model._c)
        print("Updating model")

        for _ in range(3):
            print(len(self.actor.get_batch(wait_seconds=1)))

        self.actor.stop()


def main():
    game_def = """universal_poker(
        betting=nolimit,
        bettingAbstraction=fullgame,
        numPlayers=6,
        blind=2 1 0 0 0 0,
        numRounds=4,
        firstPlayer=2 1 1 1,
        numSuits=4,
        numRanks=13,
        numHoleCards=2,
        numBoardCards=0 3 1 1,
        stack=200 200 200 200 200 200
    )
    """.replace("    ", "").replace("\n", "")
    print(game_def)

    rnad = RNad(game_def)
    rnad.step()


if __name__ == '__main__':
    main()
