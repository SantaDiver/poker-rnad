#include "RNadBot.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"


RNadBot::RNadBot(int seed, torch::jit::Module * model_)
    : rng(seed)
    , model(model_)
{
}

open_spiel::ActionsAndProbs RNadBot::GetPolicy(const open_spiel::State& state) {
    torch::Tensor info_state_tensor = torch::from_blob(
        state.InformationStateTensor().data(),
        {static_cast<long>(state.GetGame()->InformationStateTensorShape()[0])},
        torch::dtype(torch::kFloat32)
    );
    std::vector<torch::jit::IValue> model_inputs = {info_state_tensor};
    auto output = model->forward(model_inputs).toTuple()->elements();
    auto model_policy = output[output.size() - 1].toTensor();

    open_spiel::ActionsAndProbs policy;
    const std::vector<open_spiel::Action> legal_actions = state.LegalActions();
    for (open_spiel::Action action : legal_actions) {
        double prob = model_policy[action].item<double>();
        if (prob > 1e-10) policy.emplace_back(action, prob);
    }

    if (policy.empty()) {
        for (open_spiel::Action action : legal_actions) {
            policy.emplace_back(action, 1.0);
        }
    }

    open_spiel::NormalizePolicy(&policy);
    return policy;
}
