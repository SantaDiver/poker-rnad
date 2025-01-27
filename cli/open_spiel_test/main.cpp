#include "RNadBot.h"

#include <torch/script.h>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/games/universal_poker/universal_poker.h"
#include <memory>

using open_spiel::GameParameter;
using open_spiel::GameParameters;
using namespace open_spiel::universal_poker;
using namespace open_spiel::universal_poker::acpc_cpp;


int main(int argc, char** argv) {
    open_spiel::GameParameters gameParams = {
        {"betting", GameParameter("nolimit")},
        {"numPlayers", GameParameter(6)},
        {"numRounds", GameParameter(4)},
        {"blind", GameParameter("1 2 0 0 0 0")},
        {"firstPlayer", GameParameter("2 1 1 1,")},
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
    auto rNadBot = open_spiel::LoadBot("rnad", game, 0, {});

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
            auto action = rNadBot->Step(*state.get());
            std::cerr << "chose action: " << state->ActionToString(state->CurrentPlayer(), action) << std::endl;
            state->ApplyAction(action);
        }
    }

    open_spiel::Player player = 0;
    for (auto r : state->Returns()) {
        std::cerr << "Player " << player << " return " << r << std::endl;
        ++player;
    }

    return 0;
}
