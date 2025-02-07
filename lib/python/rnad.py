from typing import Dict

import pyspiel
import torch
from torch import nn
import torch.nn.functional as F

import vtrace
from poker_rnad_py import Actor


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
        policy = torch.nn.functional.normalize(exp_logits, dim=-1, p=1.)
        log_sum = torch.log(torch.sum(exp_logits, dim=-1, keepdim=True))
        log_policy = torch.where(legal_actions, logits - log_sum, 0)

        value = self.value_head(embedding)

        return (logits, log_policy, policy, value)


class RNaD:
    def __init__(self, game_def):
        self.game = pyspiel.load_game(game_def)
        self.device = torch.device('cpu')
        self.model = self.init_model()
        self.model.train()

        self.target_model = self.init_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.reg_model = self.init_model()
        self.reg_model.load_state_dict(self.model.state_dict())
        self.reg_model_prev = self.init_model()
        self.reg_model_prev.load_state_dict(self.model.state_dict())

        self.lr = 5 * 10**-5
        self.b1_adam = 0
        self.b2_adam = 0.999
        self.epsilon_adam = 10**-8
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=[self.b1_adam, self.b2_adam],
            eps=self.epsilon_adam,
        )

        jit_model = torch.jit.script(self.model).eval()
        self.actor = Actor(
            game=self.game,
            model=jit_model._c,
            num_workers=2,
            num_worked_threads=0,
            batch_size=16,
            max_queue_capacity=16,
            device_name=str(self.device)
        )
        self.actor.run()

        self.n_discrete = 32
        self.epsilon_threshold = 0.03
        self.c_bar = 1
        self.roh_bar = 1
        self.eta = 0.2
        self.gamma_averaging = 0.001
        self.vtrace_gamma = 1

        self.m = 0
        self.n = 0
        self.total_steps = 0

    def init_model(self):
        infostate_tensor_shape = self.game.information_state_tensor_shape()[0]
        num_actions = self.game.num_distinct_actions()
        model = RNadModel(
            infostate_tensor_shape=infostate_tensor_shape,
            num_actions=num_actions,
            hidden_dim=128,
            dropout=0.1
        )
        model.to(self.device)
        return model

    def batch_from_field(self, trajectories, field, dtype):
        data = torch.stack([
            torch.stack([
                torch.tensor(getattr(state, field), dtype=dtype)
                for state in trj.states
            ])
            for trj in trajectories
        ]).T
        data.to(self.device)
        return data

    def learn(self, trajectories, alpha):
        information_state = self.batch_from_field(trajectories, 'information_state', dtype=torch.float32)
        legal_actions = self.batch_from_field(trajectories, 'legal_actions', dtype=torch.int16)

        logit, log_pi, pi, v = self.model(information_state, legal_actions)
        pi_processed = vtrace.process_policy(pi, legal_actions, self.n_discrete, self.epsilon_threshold)
        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []

        returns = torch.stack([
            torch.tensor(trj.returns)
            for trj in trajectories
        ]).to(self.device)
        action = self.batch_from_field(trajectories, 'action')
        action_oh = torch.zeros_like(legal_actions)
        action_oh[action] = 1

        valid = ~self.batch_from_field(trajectories, 'is_terminal', dtype=torch.bool)
        player_id = self.batch_from_field(trajectories, 'current_player', dtype=torch.int32)
        policy = self.batch_from_field(trajectories, 'policy', dtype=torch.float32)

        with torch.no_grad():
            _, _, _, v_target = self.target_model(information_state, legal_actions)
            _, log_pi_reg, _, _ = self.reg_model(information_state, legal_actions)
            _, log_pi_reg_prev, _, _ = self.reg_model_prev(information_state, legal_actions)
            log_policy_reg = log_pi - (alpha * log_pi_reg + (1 - alpha) * log_pi_reg_prev)

            reward = returns[:, player]
            for player in range(self.game.NumPlayers()):
                v_target_, has_played, policy_target_ = vtrace.v_trace(
                    v=v_target,
                    valid=valid,
                    player_id=player_id,
                    acting_policy=policy,
                    merged_policy=pi_processed,
                    merged_log_policy=log_policy_reg,
                    player_others=vtrace._player_others(player_id, valid, player),
                    action_oh=action_oh,
                    reward=reward,
                    player=player,
                    lambda_=1.0,
                    c=self.c_bar,
                    rho=self.roh_bar,
                    eta=self.eta,
                    gamma=self.vtrace_gamma,
                )

                v_target_list.append(v_target_)
                has_played_list.append(has_played)
                v_trace_policy_target_list.append(policy_target_)

        loss_v = vtrace.get_loss_v([v] * self.game.NumPlayers(), v_target_list, has_played_list)
        is_vector = torch.unsqueeze(torch.ones_like(valid), dim=-1)
        importance_sampling_correction = [is_vector] * self.game.NumPlayers()

        loss_nerd = vtrace.get_loss_nerd(
            logit_list=[logit] * self.game.NumPlayers(),
            policy_list=[pi_processed] * self.game.NumPlayers(),
            q_vr_list=v_trace_policy_target_list,
            valid=valid,
            player_ids=player_id,
            legal_actions=legal_actions,
            importance_sampling_correction=importance_sampling_correction,
            clip=self.neurd_clip,
            threshold=self.beta,
        )

        loss = self.value_weight * loss_v + self.neurd_weight * loss_nerd
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

    def update_actor_model(self):
        jit_model = torch.jit.script(self.model).eval()
        self.actor.update_model(jit_model)

    def run(self, max_updates):
        for _ in range(max_updates):
            print(self.total_steps)
            delta_m = 1000  # [TODO]

            trajectories = self.actor.get_batch(wait_seconds=5)

        #     while self.n < delta_m:
        #         alpha = 1 if self.n > delta_m / 2 else self.n * 2 / delta_m

        #         trajectories = self.actor.get_batch(wait_seconds=5)
        #         self.learn(trajectories=trajectories, alpha=alpha)
        #         self.optimizer.step()
        #         self.optimizer.zero_grad()

        #         model_params: Dict['str', torch.Tensor] = self.model.state_dict()
        #         target_model_params: Dict['str', torch.Tensor] = self.target_model.state_dict()
        #         for name, param in model_params.items():
        #             target_model_params[name].data.copy_(
        #                 self.gamma_averaging * param.data
        #                 + (1 - self.gamma_averaging) * target_model_params[name].data
        #             )
        #         self.target_model.load_state_dict(target_model_params)
        #         self.update_actor_model()

        #         self.n += 1
        #         self.total_steps += 1

        #     self.n = 0
        #     self.m += 1
        #     self.reg_model_prev.load_state_dict(self.reg_model.state_dict())
        #     self.reg_model.load_state_dict(self.target_model.state_dict())
