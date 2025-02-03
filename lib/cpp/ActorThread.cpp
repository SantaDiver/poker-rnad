#include "ActorThread.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/universal_poker.h"
#include "open_spiel/spiel.h"
#include <exception>
#include <memory>
#include <mutex>
#include <ostream>
#include <span>
#include <thread>


namespace PokerRnaD {
    thread_local std::mt19937 rng;
}

ActorThread::ActorThread(
        const open_spiel::Game * game_,
        torch::jit::Module & model_,
        size_t num_threads
)
    : game(game_)
    , model(model_)
    , thread_pool(num_threads)
{
    initial_state = game->NewInitialState();
    playChance(initial_state);
}

std::vector<Trajectory> ActorThread::generateTrajectoriesBatch(size_t num_trajectories) const {
    StateVector state_vec(num_trajectories);

    thread_pool.detach_blocks(0, num_trajectories,
        [this, &state_vec](const std::size_t start, const std::size_t end) {
            for (size_t i = start; i < end; ++i) {
                state_vec[i] = game->NewInitialState();
                playChance(state_vec[i]);
            }
        });
    thread_pool.wait();

    std::vector<Trajectory> trajectories_vec(num_trajectories);
    while (true) {
        auto model_inputs = makeModelInputs(state_vec);
        auto output = model.forward(model_inputs).toTuple()->elements();
        auto probs = output[output.size() - 1].toTensor();

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

void ActorThread::playChance(ActorThread::StatePtr & state) {
    while (state->IsChanceNode()) {
        open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
        open_spiel::Action action = open_spiel::SampleAction(outcomes, PokerRnaD::rng).first;
        state->ApplyAction(action);
    }
}

bool ActorThread::applyAction(
        ActorThread::StateVector & state_vec,
        const ActorThread::PolicyVector & policy_vec,
        const ActorThread::ActionVector & action_vec,
        std::vector<Trajectory> & trajectories_vec
) const {
    BS::multi_future<bool> loop_future = thread_pool.submit_blocks(
        0, state_vec.size(),
        [this, &state_vec, &policy_vec, &action_vec, &trajectories_vec]
        (const std::size_t start, const std::size_t end) {
            bool has_non_terminal = false;
            for (size_t i = start; i < end; ++i) {
                bool is_terminal = state_vec[i]->IsTerminal();
                trajectories_vec[i].states.push_back(Trajectory::State{
                    .information_state = infoStateVector(state_vec[i]),
                    .current_player = state_vec[i]->CurrentPlayer(),
                    .legal_actions = legalActions(state_vec[i]),
                    .is_terminal = is_terminal,
                    .policy = policy_vec[i],
                    .action = action_vec[i]
                });
                if (is_terminal) {
                    trajectories_vec[i].returns = state_vec[i]->Returns();
                }
                else {
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

inline std::vector<float> ActorThread::infoStateVector(const ActorThread::StatePtr & state) const {
    if (state->IsTerminal()) return initial_state->InformationStateTensor();
    return state->InformationStateTensor();
}

inline ActorThread::ActionVector ActorThread::legalActions(const ActorThread::StatePtr & state) const {
    if (state->IsTerminal()) return initial_state->LegalActions();
    return state->LegalActions();
}

std::vector<torch::jit::IValue> ActorThread::makeModelInputs(const ActorThread::StateVector & state_vec) const {
    std::vector<torch::Tensor> info_state_tensor_vec(state_vec.size());
    thread_pool.detach_blocks(0, state_vec.size(),
        [this, &state_vec, &info_state_tensor_vec](const std::size_t start, const std::size_t end) {
            for (size_t i = start; i < end; ++i) {
                auto info_state_tensor = infoStateVector(state_vec[i]);
                auto tensor = torch::from_blob(
                    info_state_tensor.data(),
                    {static_cast<int64_t>(info_state_tensor.size())},
                    torch::dtype(torch::kFloat32)
                );
                info_state_tensor_vec[i] = std::move(tensor.clone());
            }
        });
    thread_pool.wait();
    return {torch::stack(info_state_tensor_vec)};
}

ActorThread::PolicyActionVectors ActorThread::sampleAction(
    const ActorThread::StateVector & state_vec, const torch::Tensor & probs) const {
    PolicyVector policy_vec(state_vec.size());
    ActionVector action_vec(state_vec.size());
    thread_pool.detach_blocks(0, state_vec.size(),
        [this, &state_vec, &probs, &policy_vec, &action_vec]
        (const std::size_t start, const std::size_t end) {
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
                const auto action = open_spiel::SampleAction(
                    policy, PokerRnaD::rng).first;
                policy_vec[i] = std::move(policy);
                action_vec[i] = action;
            }
        });
    thread_pool.wait();
    return {policy_vec, action_vec};
}
