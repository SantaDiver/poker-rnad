from typing import Dict

import pyspiel
from open_spiel.python.bots.uniform_random import UniformRandomBot

import numpy as np
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
            batch_size=256,
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
        self.neurd_clip = 1000
        self.beta = 2  # logit_clip
        self.value_weight = 1
        self.neurd_weight = 1
        self.grad_clip = 1000
        self.bounds = [100, 165, 200]
        self.delta_m = [10_000, 100_000, 35_000]

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
        ])
        data.to(self.device)
        return data

    def learn(self, trajectories, alpha):
        information_state = self.batch_from_field(trajectories, 'information_state', dtype=torch.float32)
        legal_actions = self.batch_from_field(trajectories, 'legal_actions', dtype=torch.bool)

        logit, log_pi, pi, v = self.model(information_state, legal_actions)
        pi_processed = vtrace.process_policy(pi, legal_actions, self.n_discrete, self.epsilon_threshold)
        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []

        returns = torch.stack([
            torch.tensor(trj.returns, dtype=torch.float32)
            for trj in trajectories
        ]).to(self.device)
        action = self.batch_from_field(trajectories, 'action', dtype=torch.long)
        action_oh = F.one_hot(action, num_classes=legal_actions.shape[-1])

        valid = 1 - self.batch_from_field(trajectories, 'is_terminal', dtype=torch.int16)
        player_id = self.batch_from_field(trajectories, 'current_player', dtype=torch.int32)
        policy = self.batch_from_field(trajectories, 'policy', dtype=torch.float32)

        with torch.no_grad():
            _, _, _, v_target = self.target_model(information_state, legal_actions)
            _, log_pi_reg, _, _ = self.reg_model(information_state, legal_actions)
            _, log_pi_reg_prev, _, _ = self.reg_model_prev(information_state, legal_actions)
            log_policy_reg = log_pi - (alpha * log_pi_reg + (1 - alpha) * log_pi_reg_prev)

            for player in range(self.game.num_players()):
                reward = returns[:, player]
                v_target_, has_played, policy_target_ = vtrace.v_trace(
                    v=v_target,
                    valid=valid,
                    player_id=player_id,
                    acting_policy=policy,
                    merged_policy=pi_processed,
                    merged_log_policy=log_policy_reg,
                    player_others=vtrace._player_others(player_id, valid, player),
                    actions_oh=action_oh,
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

        loss_v = vtrace.get_loss_v([v] * self.game.num_players(), v_target_list, has_played_list)
        is_vector = torch.unsqueeze(torch.ones_like(valid), dim=-1)
        importance_sampling_correction = [is_vector] * self.game.num_players()

        loss_nerd = vtrace.get_loss_nerd(
            logit_list=[logit] * self.game.num_players(),
            policy_list=[pi_processed] * self.game.num_players(),
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
        self.actor.update_model(jit_model._c)

    def get_update_info(self) -> tuple[bool, int]:

        """
        The bool value is whether the run is finished, and second is the new delta_m value
        which determines when how many steps until the next update.
        """

        bounding_indices = [i for i, bound in enumerate(self.bounds) if bound > self.m]
        if not bounding_indices:
            return False, 0

        idx = min(bounding_indices)
        return True, self.delta_m[idx]

    def print_strat(self):
        state = self.game.new_initial_state()
        cards_dealt = 0
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            if cards_dealt // 2 == 0:
                max_card = max(outcomes, key=lambda x: x[0])[0]
                state.apply_action(max_card)
            else:
                state.apply_action(outcomes[0][0])

            cards_dealt += 1

        information_state = torch.tensor([state.information_state_tensor()], dtype=torch.float32, device=self.device)
        legal_actions = torch.zeros((1, self.game.num_distinct_actions()), dtype=torch.bool, device=self.device)
        legal_actions_int = torch.tensor(state.legal_actions(), dtype=torch.int32, device=self.device)
        legal_actions[0, legal_actions_int] = True

        _, _, pi, v = self.model(information_state, legal_actions)
        print(pi)


    def play_chance(self, state):
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)


    def play_against_random(self, num_plays):
        bots = [UniformRandomBot(player, np.random) for player in range(self.game.num_players())]
        reward = 0.
        for _ in range(num_plays):
            for player in range(self.game.num_players()):
                state = self.game.new_initial_state()
                self.play_chance(state)
                while not state.is_terminal():
                    if state.current_player() == player:
                        information_state = torch.tensor([state.information_state_tensor()], dtype=torch.float32, device=self.device)
                        legal_actions = torch.zeros((1, self.game.num_distinct_actions()), dtype=torch.bool, device=self.device)
                        legal_actions_int = torch.tensor(state.legal_actions(), dtype=torch.int32, device=self.device)
                        legal_actions[0, legal_actions_int] = True

                        _, _, pi, _ = self.model(information_state, legal_actions)
                        pi = pi.detach().numpy()[0]
                        action = np.random.choice(list(range(state.num_distinct_actions())), p=pi)
                        state.apply_action(action)
                    else:
                        bot = bots[state.current_player()]
                        action = bot.step(state)
                        state.apply_action(action)

                    self.play_chance(state)

                reward += state.returns()[player]

            reward /= self.game.num_players()

        return reward / num_plays


    def run(self, max_updates):
        for _ in range(max_updates):
            may_resume, delta_m = self.get_update_info()
            if not may_resume:
                return

            while self.n < delta_m:
                alpha = 1 if self.n > delta_m / 2 else self.n * 2 / delta_m

                trajectories = self.actor.get_batch(wait_seconds=5)
                self.learn(trajectories=trajectories, alpha=alpha)
                self.optimizer.step()
                self.optimizer.zero_grad()

                model_params: Dict['str', torch.Tensor] = self.model.state_dict()
                target_model_params: Dict['str', torch.Tensor] = self.target_model.state_dict()
                for name, param in model_params.items():
                    target_model_params[name].data.copy_(
                        self.gamma_averaging * param.data
                        + (1 - self.gamma_averaging) * target_model_params[name].data
                    )
                self.target_model.load_state_dict(target_model_params)
                self.update_actor_model()

                self.n += 1
                self.total_steps += 1

                if self.total_steps > 0 and self.total_steps % 100 == 0:
                    print(self.play_against_random(1000))

            self.n = 0
            self.m += 1
            self.reg_model_prev.load_state_dict(self.reg_model.state_dict())
            self.reg_model.load_state_dict(self.target_model.state_dict())
