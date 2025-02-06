import pyspiel
import torch
from torch import nn
import torch.nn.functional as F

import vtrace


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
            hidden_dim=128,
            dropout=0.1
        )
        return model

    def batch_from_field(self, trajectories, field):
        return torch.stack([
            torch.stack([
                torch.tensor(getattr(state, field))
                for state in trj.states
            ])
            for trj in trajectories
        ]).T

    def forward(self, trajectories, alpha):
        information_state = self.batch_from_field(trajectories, 'information_state')
        legal_actions = self.batch_from_field(trajectories, 'legal_actions')

        logit, log_pi, pi, v = self.model(information_state, legal_actions)
        pi_processed = vtrace.process_policy(pi, legal_actions, self.n_discrete, self.epsilon_threshold)
        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []

        returns = torch.stack([
            torch.tensor(trj.returns)
            for trj in trajectories
        ])
        action = self.batch_from_field(trajectories, 'action')
        action_oh = torch.zeros_like(legal_actions)
        action_oh[action] = 1

        valid = ~self.batch_from_field(trajectories, 'is_terminal')
        player_id = self.batch_from_field(trajectories, 'current_player')
        policy = self.batch_from_field(trajectories, policy)

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
