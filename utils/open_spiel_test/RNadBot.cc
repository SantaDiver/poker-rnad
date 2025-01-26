#include "RNadBot.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"


RNadPolicy::RNadPolicy()
    : model(torch::jit::load("/Users/penguin-diver/personal/poker-rnad/notebooks/model.pt"))
{
    model.eval();
}

open_spiel::ActionsAndProbs RNadPolicy::GetStatePolicy(const open_spiel::State &state) const {
    torch::Tensor info_state_tensor = torch::from_blob(
        state.InformationStateTensor().data(),
        {static_cast<long>(state.GetGame()->InformationStateTensorShape()[0])},
        torch::dtype(torch::kFloat32)
    );
    std::vector<torch::jit::IValue> model_inputs = {info_state_tensor};
    auto output = model.forward(model_inputs).toTuple()->elements();
    auto model_policy = output[output.size() - 1].toTensor();

    open_spiel::ActionsAndProbs policy;
    const std::vector<open_spiel::Action> legal_actions = state.LegalActions();
    for (open_spiel::Action action : legal_actions) {
        double prob = model_policy[action].item<double>();
        if (prob > 1e-10) policy.emplace_back(action, prob);
    }

    if (policy.empty()) {
        SPIEL_DCHECK_TRUE(absl::c_find(legal_actions, ActionType::kCall) != legal_actions.end());
        policy.push_back({static_cast<open_spiel::Action>(ActionType::kCall), 1.});
    }

    if (policy.size() > 1) open_spiel::NormalizePolicy(&policy);
    return policy;
}

class RNadBotFactory : public open_spiel::BotFactory {
    bool CanPlayGame(const open_spiel::Game &game, open_spiel::Player player_id) const override {
        return absl::StrContains(game.GetType().short_name, "poker");
    }

    std::unique_ptr<open_spiel::Bot> Create(
            std::shared_ptr<const open_spiel::Game> game,
            open_spiel::Player player_id,
            const open_spiel::GameParameters & bot_params
    ) const override {
        auto policy = std::make_shared<RNadPolicy>();
        return MakePolicyBot(/*seed=*/0, policy);
    }
};

open_spiel::REGISTER_SPIEL_BOT("rnad", RNadBotFactory);
