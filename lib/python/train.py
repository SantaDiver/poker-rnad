from rnad import RNaD

def main():
    game_def = """universal_poker(
        betting=nolimit,
        bettingAbstraction=fchpa,
        numPlayers=2,
        blind=1 2,
        numRounds=4,
        firstPlayer=1 2 2 2,
        numSuits=4,
        numRanks=13,
        numHoleCards=2,
        numBoardCards=0 3 1 1,
        stack=200 200
    )
    """.replace("    ", "").replace("\n", "")
    print(game_def)

    rnad = RNaD(game_def)
    rnad.run(10)


if __name__ == '__main__':
    main()
