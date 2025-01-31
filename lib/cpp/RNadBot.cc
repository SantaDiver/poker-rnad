#include "RNadBot.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"


RNadPolicy::RNadPolicy()
{
}

open_spiel::ActionsAndProbs RNadPolicy::GetStatePolicy(const open_spiel::State &state) const {
    open_spiel::ActionsAndProbs policy;
    const std::vector<open_spiel::Action> legal_actions = state.LegalActions();

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
