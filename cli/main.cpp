#include "ActorThread.h"
#include "open_spiel/spiel.h"

int main(int, char**) {
    torch::jit::Module module;
    auto game = open_spiel::LoadGame(
        "universal_poker(" \
        "betting=nolimit," \
        "bettingAbstraction=fullgame," \
        "numPlayers=6," \
        "blind=2 1 0 0 0 0," \
        "numRounds=4," \
        "firstPlayer=2 1 1 1," \
        "numSuits=4," \
        "numRanks=13," \
        "numHoleCards=2," \
        "numBoardCards=0 3 1 1," \
        "stack=200 200 200 200 200 200)");
    ActorThread actor_thread(game.get(), module, 6);

    for (size_t i = 0; i < 100; ++i)
        actor_thread.generateTrajectoriesBatch(8);

    return 0;
}