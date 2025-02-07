from rnad import RNaD

def main():
    game_def = """universal_poker(
        betting=nolimit,
        bettingAbstraction=fullgame,
        numPlayers=6,
        blind=2 1 0 0 0 0,
        numRounds=4,
        firstPlayer=2 1 1 1,
        numSuits=4,
        numRanks=13,
        numHoleCards=2,
        numBoardCards=0 3 1 1,
        stack=200 200 200 200 200 200
    )
    """.replace("    ", "").replace("\n", "")
    print(game_def)

    rnad = RNaD(game_def)
    rnad.run(10)


if __name__ == '__main__':
    main()
