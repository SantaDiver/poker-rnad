#pragma once

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "ATen/core/TensorBody.h"
#include <memory>
#include <torch/script.h>
#include "BS_thread_pool.hpp"
#include "Trajectory.h"


class ActorThread {
private:
    static constexpr float EPS = 1e-10;

public:
    ActorThread(const open_spiel::Game * game_, torch::jit::Module & model_, size_t num_threads);

    std::vector<Trajectory> generateTrajectoriesBatch(size_t num_trajectories) const;
private:
    using StatePtr = std::unique_ptr<open_spiel::State>;
    using StateVector = std::vector<StatePtr>;
    using ActionVector = std::vector<open_spiel::Action>;
    using PolicyVector = std::vector<open_spiel::ActionsAndProbs>;
    using PolicyActionVectors = std::pair<PolicyVector, ActionVector>;

    [[nodiscard]] bool applyAction(
        ActorThread::StateVector & state_vec,
        const ActorThread::PolicyVector & policy_vec,
        const ActorThread::ActionVector & action_vec,
        std::vector<Trajectory> & trajectories_vec
    ) const;
    std::vector<torch::jit::IValue> makeModelInputs(const ActorThread::StateVector & state_vec) const;
    PolicyActionVectors sampleAction(const ActorThread::StateVector & state_vec, const torch::Tensor & probs) const;

    static void playChance(StatePtr & state);
    std::vector<float> infoStateVector(const StatePtr & state) const;
    ActionVector legalActions(const StatePtr & state) const;

    const open_spiel::Game * game;
    mutable torch::jit::Module model;
    mutable BS::light_thread_pool thread_pool;
    StatePtr initial_state;
};