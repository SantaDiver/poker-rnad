import torch
from poker_rnad import RNaD


def main():
    # game_def = "leduc_poker(players=2)"
    game_def = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 50,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,"
        "numBoardCards=0 3 1 1,stack=10000 10000,bettingAbstraction=fchpa)"
    )
    print(game_def)

    rnad = RNaD(
        game_def=game_def,
        num_workers=2,
        num_threads=12,
        batch_size=512,
        max_queue_capacity=4,
        hidden_dim=256,
        device=torch.device('cpu')
    )
    print("built rnad")
    rnad.run(1000)


if __name__ == '__main__':
    main()
