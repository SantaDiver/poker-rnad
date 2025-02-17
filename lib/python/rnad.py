from typing import Dict
import time

import pyspiel
from open_spiel.python.bots.uniform_random import UniformRandomBot
from open_spiel.python.algorithms import exploitability

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import vtrace
from .nn import MLP
from poker_rnad_py import Actor
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class EnvStep:
  """Holds the tensor data representing the current game state."""
  # Indicates whether the state is a valid one or just a padding. Shape: [...]
  # The terminal state being the first one to be marked !valid.
  # All other tensors in EnvStep contain data, but only for valid timesteps.
  # Once !valid the data needs to be ignored, since it's a duplicate of
  # some other previous state.
  # The rewards is the only exception that contains reward values
  # in the terminal state, which is marked !valid.
  # TODO(author16): This is a confusion point and would need to be clarified.
  valid: torch.Tensor  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The single tensor representing the state observation. Shape: [..., ??]
  obs: torch.Tensor  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The legal actions mask for the current player. Shape: [..., A]
  legal: torch.Tensor  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The current player id as an int. Shape: [...]
  player_id: torch.Tensor  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The rewards of all the players. Shape: [..., P]
  rewards: torch.Tensor  # pytype: disable=annotation-type-mismatch  # numpy-scalars


@dataclass(frozen=True)
class ActorStep:
  """The actor step tensor summary."""
  # The action (as one-hot) of the current player. Shape: [..., A]
  action_oh: torch.Tensor  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The policy of the current player. Shape: [..., A]
  policy: torch.Tensor  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The rewards of all the players. Shape: [..., P]
  # Note - these are rewards obtained *after* the actor step, and thus
  # these are the same as EnvStep.rewards visible before the *next* step.
  rewards: torch.Tensor  # pytype: disable=annotation-type-mismatch  # numpy-scalars


@dataclass(frozen=True)
class TimeStep:
  """The tensor data for one game transition (env_step, actor_step)."""
  env: EnvStep
  actor: ActorStep


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
        self.num_actions = num_actions

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

        legal_actions = legal_actions.to(torch.bool)

        logits = self.policy_head(embedding)
        exp_logits = torch.where(legal_actions, torch.exp(logits), 0)
        policy = torch.nn.functional.normalize(exp_logits, dim=-1, p=1.)
        log_sum = torch.log(torch.sum(exp_logits, dim=-1, keepdim=True))
        log_policy = torch.where(legal_actions, logits - log_sum, 0)

        value = self.value_head(embedding)

        return (logits, log_policy, policy, value)


class ModelPolicy(pyspiel.Policy):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

    def action_probabilities(self, state, player_id=None):
        information_state = torch.tensor([state.information_state_tensor()], dtype=torch.float32, device=self.device)
        legal_actions = torch.zeros((1, state.get_game().num_distinct_actions()), dtype=torch.bool, device=self.device)
        legal_actions_int = torch.tensor(state.legal_actions(), dtype=torch.int32, device=self.device)
        legal_actions[0, legal_actions_int] = True

        _, _, pi, _ = self.model(information_state, legal_actions)
        pi = pi.detach().numpy()[0]

        return {
            action : pi[action]
            for action in state.legal_actions()
        }


class RNaD:
    def __init__(
            self,
            game_def,
            num_workers,
            num_threads,
            batch_size,
            max_queue_capacity,
            device=torch.device('cpu')
    ):
        self.game_def = game_def
        self.batch_size = batch_size
        self.game = pyspiel.load_game(game_def)
        self.device = device
        self.model = self.init_model()
        self.model.train()

        self._np_rng = np.random.RandomState(42)
        self._ex_state = self.play_chance(self.game.new_initial_state())

        self.target_model = self.init_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.reg_model = self.init_model()
        self.reg_model.load_state_dict(self.model.state_dict())
        self.reg_model_prev = self.init_model()
        self.reg_model_prev.load_state_dict(self.model.state_dict())

        self.lr = 5e-5
        self.b1_adam = 0
        self.b2_adam = 0.999
        self.epsilon_adam = 10e-8
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=[self.b1_adam, self.b2_adam],
            eps=self.epsilon_adam,
        )

        # self.actor = Actor(
        #     game=self.game_def,
        #     model=self.build_jit_model()._c,
        #     num_workers=num_workers,
        #     num_threads=num_threads,
        #     batch_size=batch_size,
        #     max_queue_capacity=max_queue_capacity,
        #     device_name=str(self.device)
        # )
        # self.actor.run()

        self.n_discrete = 32
        self.epsilon_threshold = 0.03
        self.c_bar = 1.
        self.roh_bar = np.inf
        self.eta = 0.2
        self.gamma_averaging = 0.001
        self.vtrace_gamma = 1
        self.neurd_clip = 10000
        self.beta = 2.  # logit_clip
        self.value_weight = 1
        self.neurd_weight = 1
        self.grad_clip = 10000
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
            hidden_dim=16,
            dropout=0.0
        )
        model = model.to(self.device)
        return model

    def learn(self, trajectories, alpha):
        # information_state = trajectories.information_state
        information_state = trajectories.env.obs

        # legal_actions = trajectories.legal_actions
        legal_actions = trajectories.env.legal

        # returns = trajectories.returns
        returns = trajectories.actor.rewards

        # action = trajectories.action
        # action_oh = F.one_hot(action, num_classes=legal_actions.shape[-1])
        action_oh = trajectories.actor.action_oh

        # valid = 1 - trajectories.is_terminal
        valid = trajectories.env.valid

        # player_id = trajectories.current_player
        player_id = trajectories.env.player_id

        # policy = trajectories.policy
        policy = trajectories.actor.policy

        logit, log_pi, pi, v = self.model(information_state, legal_actions)
        pi[...] = 0.
        pi[:, :, -1] = 1.
        pi_processed = vtrace.process_policy(pi, legal_actions, self.n_discrete, self.epsilon_threshold)

        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
        with torch.no_grad():
            _, _, _, v_target = self.target_model(information_state, legal_actions)
            _, log_pi_reg, _, _ = self.reg_model(information_state, legal_actions)
            _, log_pi_reg_prev, _, _ = self.reg_model_prev(information_state, legal_actions)
            log_policy_reg = log_pi - (alpha * log_pi_reg + (1 - alpha) * log_pi_reg_prev)

            for player in range(self.game.num_players()):
                reward = returns[..., player]
                # print(returns[..., player])
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

                # print(v_target_)

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

    def build_jit_model(self):
        model_copy = self.init_model()
        model_copy.load_state_dict(self.model.state_dict())
        jit_model = torch.jit.script(model_copy)
        jit_model.eval()
        return jit_model

    def update_actor_model(self):
        self.actor.update_model(self.build_jit_model()._c)

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

    def play_chance(self, state):
        while state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = self._np_rng.choice(action_list, p=prob_list)
            state.apply_action(action)

        return state

    def play_against_random(self, num_plays):
        bots = [UniformRandomBot(player, np.random) for player in range(self.game.num_players())]
        reward = 0.
        for _ in range(num_plays):
            for player in range(self.game.num_players()):
                state = self.game.new_initial_state()
                state = self.play_chance(state)
                while not state.is_terminal():
                    if state.current_player() == player:
                        information_state = torch.tensor(state.information_state_tensor(), dtype=torch.float32, device=self.device)
                        legal_actions = torch.zeros((self.game.num_distinct_actions(), ), dtype=torch.bool, device=self.device)
                        legal_actions_int = torch.tensor(state.legal_actions(), dtype=torch.int32, device=self.device)
                        legal_actions[legal_actions_int] = True

                        _, _, pi, _ = self.model(information_state, legal_actions)
                        pi = pi.cpu().detach().numpy()
                        action = np.random.choice(list(range(state.num_distinct_actions())), p=pi)
                        state.apply_action(action)
                    else:
                        bot = bots[state.current_player()]
                        action = bot.step(state)
                        state.apply_action(action)

                    state = self.play_chance(state)

                reward += state.returns()[player]

        return reward / (self.game.num_players() * num_plays)

    def nash_conv(self):
        jit_model = torch.jit.script(self.model).eval()
        model_policy = ModelPolicy(model=jit_model, device=self.device)
        return exploitability.exploitability(self.game, model_policy)

    def actor_step(self, env_step: EnvStep):
        _, _, pi, _ = self.model(env_step.obs, env_step.legal)
        pi = pi.detach().cpu()
        pi = np.asarray(pi).astype("float64")
        # TODO(author18): is this policy normalization really needed?
        pi = pi / np.sum(pi, axis=-1, keepdims=True)

        pi[...] = 0.
        pi[:, -1] = 1.

        action = np.apply_along_axis(
        lambda x: self._np_rng.choice(range(pi.shape[1]), p=x), axis=-1, arr=pi)
        # TODO(author16): reapply the legal actions mask to bullet-proof sampling.
        action_oh = np.zeros(pi.shape, dtype="float64")
        action_oh[range(pi.shape[0]), action] = 1.0

        actor_step = ActorStep(
            policy=torch.from_numpy(pi),
            action_oh=torch.from_numpy(action_oh),
            rewards=torch.empty([])
        )  # pytype: disable=wrong-arg-types  # numpy-scalars

        return action, actor_step

    def collect_batch_trajectory(self) -> TimeStep:
        states = [
            self.play_chance(self.game.new_initial_state())
            for _ in range(self.batch_size)
        ]
        timesteps = []

        env_step = self._batch_of_states_as_env_step(states)
        trajectory_max = 10
        for _ in range(trajectory_max):
            prev_env_step = env_step
            a, actor_step = self.actor_step(env_step)

            states = self._batch_of_states_apply_action(states, a)
            env_step = self._batch_of_states_as_env_step(states)
            timesteps.append(
                TimeStep(
                    env=prev_env_step,
                    actor=ActorStep(
                        action_oh=actor_step.action_oh,
                        policy=actor_step.policy,
                        rewards=env_step.rewards),
                ))
        # Concatenate all the timesteps together to form a single rollout [T, B, ..]
        return TimeStep(
            actor=ActorStep(**{
                field.name: torch.stack(list(map(lambda t: getattr(t.actor, field.name), timesteps)), dim=0)
                for field in fields(ActorStep)
            }),
            env=EnvStep(**{
                field.name: torch.stack(list(map(lambda t: getattr(t.env, field.name), timesteps)), dim=0)
                for field in fields(EnvStep)
            })
        )

    def _state_as_env_step(self, state: pyspiel.State) -> EnvStep:
        # A  state must be communicated to players, however since
        # it's a terminal state things like the state_representation or
        # the set of legal actions are meaningless and only needed
        # for the sake of creating well a defined trajectory tensor.
        # Therefore the code below:
        # - extracts the rewards
        # - if the state is terminal, uses a dummy other state for other fields.
        rewards = torch.tensor(state.returns(), dtype=torch.float32)

        valid = not state.is_terminal()
        if not valid:
            state = self._ex_state

        obs = state.information_state_tensor()

        # TODO(author16): clarify the story around rewards and valid.
        return EnvStep(
            obs=torch.tensor(obs, dtype=torch.float32),
            legal=torch.tensor(state.legal_actions_mask(), dtype=torch.int8),
            player_id=torch.tensor(state.current_player(), dtype=torch.float32),
            valid=torch.tensor(valid, dtype=torch.float32),
            rewards=rewards)

    def _batch_of_states_as_env_step(self, states) -> EnvStep:
        envs = [self._state_as_env_step(state) for state in states]
        return EnvStep(**{
            field.name: torch.stack(list(map(lambda e: getattr(e, field.name), envs)), dim=0)
            for field in fields(EnvStep)
        })

    def _batch_of_states_apply_action(self, states, actions):
        assert len(states) == len(actions)
        for state, action in zip(states, list(actions)):
            if not state.is_terminal():
                state.apply_action(action)
                self.play_chance(state)
        return states

    def run(self, max_updates):
        start = time.time()
        for _ in range(max_updates):
            may_resume, delta_m = self.get_update_info()
            if not may_resume:
                return

            while self.n < delta_m:
                alpha = 1 if self.n > delta_m / 2 else self.n * 2 / delta_m

                # trajectories = self.actor.get_batch(wait_seconds=5)
                trajectories = self.collect_batch_trajectory()

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
                # self.update_actor_model()

                self.n += 1
                self.total_steps += 1

                # if self.total_steps > 0 and self.total_steps % 1000 == 0:
                #     print("win against random: ", self.play_against_random(10000))
                #     print("expl best response: ", self.nash_conv())
                # elif self.total_steps > 0 and self.total_steps % 100 == 0:
                #     end = time.time()
                #     print(self.total_steps, "time elapsed: ", end - start)
                #     start = end

                if self.total_steps > 0 and self.total_steps % 100 == 0:
                    print("expl best response: ", self.nash_conv())

            self.n = 0
            self.m += 1
            self.reg_model_prev.load_state_dict(self.reg_model.state_dict())
            self.reg_model.load_state_dict(self.target_model.state_dict())
