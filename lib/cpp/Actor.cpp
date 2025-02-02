#include "Actor.h"

Actor::Actor(const open_spiel::Game & game_, torch::jit::Module & model_)
    : game(game_)
    , model(model_)
    , rng(absl::ToUnixNanos(absl::Now()))
{
    initial_state = game.NewInitialState();
    playChance(initial_state);
}

Actor::TrajectoriesVector Actor::generateTrajectoriesBatch(size_t num_trajectories) const {
    StateVector state_vec;
    state_vec.reserve(num_trajectories);
    for (size_t i = 0; i < num_trajectories; ++i) {
        state_vec.push_back(game.NewInitialState());
        playChance(state_vec[i]);
    }

    TrajectoriesVector trajectories_vec(num_trajectories);
    while (true) {
        const bool all_terminal = std::all_of(state_vec.begin(), state_vec.end(),
            [](const auto & s) { return s->IsTerminal(); });
        if (all_terminal) break;

        auto [policy_vec, action_vec] = applyModel(state_vec);
        updateTrajectories(state_vec, policy_vec, action_vec, trajectories_vec);
        applyAction(state_vec, action_vec);
    }

    return trajectories_vec;
}

inline void Actor::playChance(Actor::StatePtr & state) const {
    while (state->IsChanceNode()) {
        open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
        open_spiel::Action action = open_spiel::SampleAction(outcomes, rng).first;
        state->ApplyAction(action);
    }
}

inline void Actor::applyAction(Actor::StateVector & state_vec, const Actor::ActionVector & action_vec) const {
    assert(state_vec.size() == action_vec.size());
    for (size_t i = 0; i < state_vec.size(); ++i) {
        if (state_vec[i]->IsTerminal()) continue;
        state_vec[i]->ApplyAction(action_vec[i]);
        playChance(state_vec[i]);
    }
}

inline std::vector<float> Actor::infoStateVector(const Actor::StatePtr & state) const {
    if (state->IsTerminal()) return initial_state->InformationStateTensor();
    return state->InformationStateTensor();
}

inline Actor::ActionVector Actor::legalActions(const Actor::StatePtr & state) const {
    if (state->IsTerminal()) return initial_state->LegalActions();
    return state->LegalActions();
}

Actor::PoliciesActions Actor::applyModel(const Actor::StateVector & state_vec) const {
    std::vector<torch::Tensor> info_state_tensor_vec;
    info_state_tensor_vec.reserve(state_vec.size());
    for (const auto & state : state_vec) {
        auto info_state_tensor = infoStateVector(state);
        auto tensor = torch::from_blob(
            info_state_tensor.data(),
            {static_cast<int64_t>(info_state_tensor.size())},
            torch::dtype(torch::kFloat32)
        );
        info_state_tensor_vec.push_back(tensor.clone());
    }
    std::vector<torch::jit::IValue> model_inputs = {torch::stack(info_state_tensor_vec)};
    auto output = model.forward(model_inputs).toTuple()->elements();
    auto policies = output[output.size() - 1].toTensor();

    PolicyVector policy_vec;
    policy_vec.reserve(state_vec.size());
    ActionVector action_vec;
    action_vec.reserve(action_vec.size());
    for (size_t i = 0; i < state_vec.size(); ++i) {
        open_spiel::ActionsAndProbs policy;
        const std::vector<open_spiel::Action> legal_actions = legalActions(state_vec[i]);
        for (open_spiel::Action action : legal_actions) {
            double prob = policies[i][action].item<double>();
            if (prob > EPS) policy.emplace_back(action, prob);
        }
        if (policy.empty()) {
            for (open_spiel::Action action : legal_actions) {
                policy.emplace_back(action, 1.0);
            }
        }
        open_spiel::NormalizePolicy(&policy);

        const auto action = open_spiel::SampleAction(policy, rng).first;
        policy_vec.push_back(std::move(policy));
        action_vec.push_back(action);
    }

    return {policy_vec, action_vec};
}

inline void Actor::updateTrajectories(
        const Actor::StateVector & state_vec,
        const Actor::PolicyVector & policy_vec,
        const Actor::ActionVector & action_vec,
        Actor::TrajectoriesVector & trajectories_vec
) const {
    for (size_t i = 0; i < trajectories_vec.size(); ++i) {
        bool is_terminal = state_vec[i]->IsTerminal();
        trajectories_vec[i].states.push_back(Trajectory::State{
            .information_state = infoStateVector(state_vec[i]),
            .current_player = state_vec[i]->CurrentPlayer(),
            .legal_actions = legalActions(state_vec[i]),
            .is_terminal = is_terminal,
            .policy = policy_vec[i],
            .action = action_vec[i]
        });
        if (is_terminal) trajectories_vec[i].returns = state_vec[i]->Returns();
    }
}
