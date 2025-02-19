#include "ActorWorker.h"

#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/universal_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <torch/types.h>
#include <c10/core/Device.h>


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
        const auto trajectories_batch = generateTrajectoriesBatch(batch_size);
        const auto trajectory_tensors = trajectoryToTensors(trajectories_batch);
        queue->Push(trajectory_tensors);
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
                const auto prev_state = state_vec[i]->Clone();
                if (!state_vec[i]->IsTerminal()) {
                    has_non_terminal = true;
                    state_vec[i]->ApplyAction(action_vec[i]);
                    playChance(state_vec[i]);
                }
                trajectories_vec[i].states.push_back(Trajectory::State{
                    .information_state = infoStateVector(prev_state),
                    .current_player = prev_state->CurrentPlayer(),
                    .legal_actions = prev_state->LegalActionsMask(),
                    .is_terminal = prev_state->IsTerminal(),
                    .policy = policy,
                    .action = action_vec[i],
                    .returns = state_vec[i]->Returns()
                });
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

                auto legal_actions_vector = state_vec[i]->LegalActionsMask();
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

TrajectoryTensors ActorWorker::trajectoryToTensors(const TrajectoryBatch & trajectories_vec) const {
    SPIEL_CHECK_GT(trajectories_vec.size(), 0);

    TrajectoryTensors result(
        trajectories_vec[0].states.size(),
        trajectories_vec.size(),
        game->InformationStateTensorSize(),
        game->NumDistinctActions(),
        game->NumPlayers()
    );

    thread_pool.detach_blocks(0, trajectories_vec.size(),
        [&trajectories_vec, &result]
        (const std::size_t start, const std::size_t end) {
            for (size_t b = start; b < end; ++b) {
                for (size_t t = 0; t < trajectories_vec[0].states.size(); ++t) {
                    result.information_state[t][b] = torch::tensor(
                        trajectories_vec[b].states[t].information_state,
                        torch::TensorOptions().dtype(torch::kFloat32)
                    );
                    result.current_player[t][b] = torch::tensor(
                        trajectories_vec[b].states[t].current_player,
                        torch::TensorOptions().dtype(torch::kInt64)
                    );
                    result.legal_actions[t][b] = torch::tensor(
                        trajectories_vec[b].states[t].legal_actions,
                        torch::TensorOptions().dtype(torch::kBool)
                    );
                    result.is_terminal[t][b] = torch::tensor(
                        trajectories_vec[b].states[t].is_terminal,
                        torch::TensorOptions().dtype(torch::kInt64)
                    );
                    result.policy[t][b] = torch::tensor(
                        trajectories_vec[b].states[t].policy,
                        torch::TensorOptions().dtype(torch::kFloat32)
                    );
                    result.action[t][b] = torch::tensor(
                        trajectories_vec[b].states[t].action,
                        torch::TensorOptions().dtype(torch::kInt64)
                    );
                    result.returns[t][b] = torch::tensor(
                        trajectories_vec[b].states[t].returns,
                        torch::TensorOptions().dtype(torch::kFloat32)
                    );
                }
            }
    });
    thread_pool.wait();

    result.information_state = result.information_state.to(device_name);
    result.current_player = result.current_player.to(device_name);
    result.legal_actions = result.legal_actions.to(device_name);
    result.is_terminal = result.is_terminal.to(device_name);
    result.policy = result.policy.to(device_name);
    result.action = result.action.to(device_name);
    result.returns = result.returns.to(device_name);

    return result;
}
