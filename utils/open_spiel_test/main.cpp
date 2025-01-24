#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/universal_poker/universal_poker.h"
#include <memory>

using open_spiel::GameParameter;
using open_spiel::GameParameters;
using namespace open_spiel::universal_poker;
using namespace open_spiel::universal_poker::acpc_cpp;


void PrintLegalActions(const ACPCState & state,
                       open_spiel::Player player) {
    std::cerr << "Legal moves for player " << player << ":" << std::endl;
    for (ACPCState::ACPCActionType action_type : {ACPCState::ACPC_FOLD, ACPCState::ACPC_CALL}) {
        std::cerr << "  " << state.IsValidAction(action_type, 0) << std::endl;
    }
}

int main(int argc, char** argv) {
    open_spiel::GameParameters gameParams = {
        {"betting", GameParameter("nolimit")},
        {"numPlayers", GameParameter(6)},
        {"blind", GameParameter("1 2 0 0 0 0")},
        {"numSuits", GameParameter(4)},
        {"numRanks", GameParameter(13)},
        {"numHoleCards", GameParameter(2)},
        {"numBoardCards", GameParameter("0 3 1 1")},
        {"stack", GameParameter("200 200 200 200 200 200")},
        {"bettingAbstraction", GameParameter("fullgame")}
    };
    auto game = std::make_shared<const UniversalPokerGame>(gameParams);

    std::mt19937 rng(time(0));

    std::cerr << "Starting new game..." << std::endl;
    std::unique_ptr<UniversalPokerState> state(
        dynamic_cast<UniversalPokerState *>(game->NewInitialState().release())
    );

    std::cerr << "Initial state:" << std::endl << state->ToString() << std::endl;

    while (!state->IsTerminal()) {
        std::cerr << "player " << state->CurrentPlayer() << std::endl;

        if (state->IsChanceNode()) {
            // Chance node; sample one according to underlying distribution.
            std::vector<std::pair<open_spiel::Action, double>> outcomes =
                state->ChanceOutcomes();
            open_spiel::Action action = open_spiel::SampleAction(outcomes, rng).first;
            std::cerr << "sampled outcome: "
                      << state->ActionToString(open_spiel::kChancePlayerId, action)
                      << std::endl;
            state->ApplyAction(action);
        } else {
            assert(!state->IsSimultaneousNode());
            // Decision node, sample one uniformly.
            auto player = state->CurrentPlayer();
            std::vector<open_spiel::Action> actions = state->LegalActions();

            const acpc_cpp::ACPCState & acpcState = state->acpc_state();
            PrintLegalActions(acpcState, player);

            absl::uniform_int_distribution<> dis(0, actions.size() - 1);
            auto action = actions[dis(rng)];
            std::cerr << "chose action: " << state->ActionToString(player, action)
                        << std::endl;
            state->ApplyAction(action);
        }
    }
}
