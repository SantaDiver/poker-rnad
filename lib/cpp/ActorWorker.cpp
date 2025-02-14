#include "ActorWorker.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/universal_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include <atomic>
#include <c10/core/Device.h>
#include <exception>
#include <memory>
#include <mutex>
#include <ostream>
#include <random>
#include <span>
#include <thread>
#include <torch/types.h>


ActorWorker::ActorWorker(
        const open_spiel::Game * game_,
        const torch::jit::Module & model_,
        BS::light_thread_pool & thread_pool_,
        Queue * queue_,
        size_t batch_size_,
        const std::string_view device_name_
)
    : game(game_)
    , model(model_)
    , thread_pool(thread_pool_)
    , queue(queue_)
    , batch_size(batch_size_)
    , device_name(device_name_)
    , running(false)
{
    initial_state = game->NewInitialState();
    playChance(initial_state);
}

void ActorWorker::run() {
    SPIEL_CHECK_TRUE(queue);
    SPIEL_CHECK_GT(batch_size, 0);

    running = true;

    while(running.load(std::memory_order_relaxed)) {
        queue->Push(generateTrajectoriesBatch(batch_size));
    }
}

void ActorWorker::stop() {
    running = false;
    queue->BlockNewValues();
}

void ActorWorker::updateModel(const torch::jit::Module & model_) {
    std::lock_guard<std::mutex> lock(mtx);
    model = model_;
}

ActorWorker::TrajectoryBatch ActorWorker::generateTrajectoriesBatch(size_t num_trajectories) const {
    StateVector state_vec(num_trajectories);

    thread_pool.detach_blocks(0, num_trajectories,
        [this, &state_vec](const std::size_t start, const std::size_t end) {
            for (size_t i = start; i < end; ++i) {
                state_vec[i] = game->NewInitialState();
                playChance(state_vec[i]);
            }
        });
    thread_pool.wait();

    TrajectoryBatch trajectories_vec(num_trajectories);
    while (true) {
        std::lock_guard<std::mutex> lock(mtx);
        auto model_inputs = makeModelInputs(state_vec);
        auto output = model.forward(model_inputs).toTuple()->elements();
        auto probs = output[2].toTensor().to(torch::kCPU).detach();

        const auto [policy_vec, action_vec] = sampleAction(state_vec, probs);

        const bool has_non_terminal = applyAction(
            state_vec,
            policy_vec,
            action_vec,
            trajectories_vec);
        if (!has_non_terminal) break;
    }

    return trajectories_vec;
}

void ActorWorker::playChance(ActorWorker::StatePtr & state) {
    std::mt19937 rng{std::random_device{}()};
    while (state->IsChanceNode()) {
        open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
        const open_spiel::Action action = open_spiel::SampleAction(outcomes, rng).first;
        state->ApplyAction(action);
    }
}

bool ActorWorker::applyAction(
        StateVector & state_vec,
        const PolicyVector & policy_vec,
        const ActionVector & action_vec,
        TrajectoryBatch & trajectories_vec
) const {
    BS::multi_future<bool> loop_future = thread_pool.submit_blocks(
        0, state_vec.size(),
        [this, &state_vec, &policy_vec, &action_vec, &trajectories_vec]
        (const std::size_t start, const std::size_t end) {
            bool has_non_terminal = false;
            for (size_t i = start; i < end; ++i) {
                std::vector<double> policy(game->NumDistinctActions(), 0.);
                for (auto [action, prob] : policy_vec[i])
                    policy[action] = prob;

                bool is_terminal = state_vec[i]->IsTerminal();
                trajectories_vec[i].states.push_back(Trajectory::State{
                    .information_state = infoStateVector(state_vec[i]),
                    .current_player = state_vec[i]->CurrentPlayer(),
                    .legal_actions = legalActionAsMask(state_vec[i]),
                    .is_terminal = is_terminal,
                    .policy = policy,
                    .action = action_vec[i],
                    .returns = state_vec[i]->Returns()
                });
                if (!is_terminal) {
                    has_non_terminal = true;
                    state_vec[i]->ApplyAction(action_vec[i]);
                    playChance(state_vec[i]);
                }
            }

            return has_non_terminal;
        });
    std::vector<bool> partial = loop_future.get();
    return std::any_of(partial.begin(), partial.end(),
        [](const bool v){ return v; });
}

inline std::vector<float> ActorWorker::infoStateVector(const StatePtr & state) const {
    if (state->IsTerminal()) return initial_state->InformationStateTensor();
    return state->InformationStateTensor();
}

inline ActorWorker::ActionVector ActorWorker::legalActions(const StatePtr & state) const {
    if (state->IsTerminal()) return initial_state->LegalActions();
    return state->LegalActions();
}

inline ActorWorker::ActionMask ActorWorker::legalActionAsMask(const StatePtr & state) const {
    ActionMask legal_actions(game->NumDistinctActions(), 0);
    for (open_spiel::Action action : legalActions(state))
        legal_actions[action] = 1;
    return legal_actions;
}

std::vector<torch::jit::IValue> ActorWorker::makeModelInputs(const StateVector & state_vec) const {
    std::vector<torch::Tensor> info_state_tensor_vec(state_vec.size());
    std::vector<torch::Tensor> legal_actions_vec(state_vec.size());

    thread_pool.detach_blocks(0, state_vec.size(),
    [this, &state_vec, &info_state_tensor_vec, &legal_actions_vec]
        (const std::size_t start, const std::size_t end) {
            for (size_t i = start; i < end; ++i) {
                auto info_state_vector = infoStateVector(state_vec[i]);
                info_state_tensor_vec[i] = torch::tensor(
                    info_state_vector,
                    torch::TensorOptions().dtype(torch::kFloat32)
                );

                auto legal_actions_vector = legalActionAsMask(state_vec[i]);
                legal_actions_vec[i] = torch::tensor(
                    legal_actions_vector,
                    torch::TensorOptions().dtype(torch::kBool)
                );
            }
        });
    thread_pool.wait();
    return {
        torch::stack(info_state_tensor_vec).to(device_name),
        torch::stack(legal_actions_vec).to(device_name)
    };
}

ActorWorker::PolicyActionVectors ActorWorker::sampleAction(
    const ActorWorker::StateVector & state_vec, const torch::Tensor & probs) const {
    PolicyVector policy_vec(state_vec.size());
    ActionVector action_vec(state_vec.size());
    thread_pool.detach_blocks(0, state_vec.size(),
        [this, &state_vec, &probs, &policy_vec, &action_vec]
        (const std::size_t start, const std::size_t end) {
            std::mt19937 rng{std::random_device{}()};
            for (size_t i = start; i < end; ++i) {
                open_spiel::ActionsAndProbs policy;
                const std::vector<open_spiel::Action> legal_actions = legalActions(state_vec[i]);
                for (open_spiel::Action action : legal_actions) {
                    double prob = probs[i][action].item<double>();
                    if (prob > EPS) policy.emplace_back(action, prob);
                }
                if (policy.empty()) {
                    for (open_spiel::Action action : legal_actions) {
                        policy.emplace_back(action, 1.0);
                    }
                }
                open_spiel::NormalizePolicy(&policy);
                action_vec[i] = open_spiel::SampleAction(policy, rng).first;
                policy_vec[i] = std::move(policy);
            }
        });
    thread_pool.wait();
    return {policy_vec, action_vec};
}
