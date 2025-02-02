#pragma once

#include <memory>
#include <string>
#include <ranges>
#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>

#include "ATen/core/TensorBody.h"
#include "absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

#include "open_spiel/spiel_utils.h"
#include "Trajectory.h"


class Actor {
    static constexpr float EPS = 1e-10;

public:
    using TrajectoriesVector = std::vector<Trajectory>;
    Actor(const open_spiel::Game & game_, torch::jit::Module & model_);
    TrajectoriesVector generateTrajectoriesBatch(size_t num_trajectories) const;

private:
    using StatePtr = std::unique_ptr<open_spiel::State>;
    using StateVector = std::vector<StatePtr>;
    using ActionVector = std::vector<open_spiel::Action>;
    using PolicyVector = std::vector<open_spiel::ActionsAndProbs>;
    using PoliciesActions = std::pair<PolicyVector, ActionVector>;

    void playChance(StatePtr & state) const;
    void applyAction(StateVector & state_vec, const ActionVector & action_vec) const;
    std::vector<float> infoStateVector(const StatePtr & state) const;
    ActionVector legalActions(const StatePtr & state) const;
    PoliciesActions applyModel(const StateVector & state_vec) const;
    void updateTrajectories(
            const StateVector & state_vec,
            const PolicyVector & policy_vec,
            const ActionVector & action_vec,
            TrajectoriesVector & trajectories_vec
    ) const;

    const open_spiel::Game & game;
    StatePtr initial_state;
    mutable torch::jit::Module model;
    mutable std::mt19937 rng;
};
