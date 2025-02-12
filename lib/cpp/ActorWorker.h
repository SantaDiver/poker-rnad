#pragma once

#include <memory>

#include "ATen/core/TensorBody.h"
#include <torch/script.h>
#include "BS_thread_pool.hpp"
#include "libfork/schedule.hpp"

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/threaded_queue.h"

#include "Trajectory.h"


class ActorWorker {
private:
    static constexpr double EPS = 1e-10;

public:
    using TrajectoryBatch = std::vector<Trajectory>;
    using Queue = open_spiel::ThreadedQueue<TrajectoryBatch>;

    ActorWorker(
        const open_spiel::Game * game_,
        const torch::jit::Module & model_,
        const lf::lazy_pool & pool_,
        Queue * queue_ = nullptr,
        size_t batch_size_ = 0,
        const std::string_view device_name_ = "cpu"
    );

    TrajectoryBatch generateTrajectoriesBatch(size_t num_trajectories) const;
    void run();
    void stop();
    void updateModel(const torch::jit::Module & model_);

private:
    using StatePtr = std::unique_ptr<open_spiel::State>;
    using StateVector = std::vector<StatePtr>;
    using ActionVector = std::vector<open_spiel::Action>;
    using PolicyVector = std::vector<open_spiel::ActionsAndProbs>;
    using PolicyActionVectors = std::pair<PolicyVector, ActionVector>;
    using ActionMask = std::vector<uint8_t>;

    [[nodiscard]] bool applyAction(
        ActorWorker::StateVector & state_vec,
        const ActorWorker::PolicyVector & policy_vec,
        const ActorWorker::ActionVector & action_vec,
        TrajectoryBatch & trajectories_vec
    ) const;
    std::vector<torch::jit::IValue> makeModelInputs(const ActorWorker::StateVector & state_vec) const;
    PolicyActionVectors sampleAction(const ActorWorker::StateVector & state_vec, const torch::Tensor & probs) const;

    static void playChance(StatePtr & state);
    std::vector<float> infoStateVector(const StatePtr & state) const;
    ActionVector legalActions(const StatePtr & state) const;
    ActionMask legalActionAsMask(const StatePtr & state) const;

    const open_spiel::Game * game;
    mutable torch::jit::Module model;
    const lf::lazy_pool & pool;
    mutable BS::light_thread_pool thread_pool;
    Queue * queue;
    const size_t batch_size;
    const std::string device_name;

    StatePtr initial_state;
    std::atomic<bool> running;
    mutable std::mutex mtx;
};