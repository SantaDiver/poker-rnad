import open_spiel_rnad as rnad
from open_spiel.python.algorithms import exploitability

def main():
    solver = rnad.RNaDSolver(rnad.RNaDConfig(game_name="kuhn_poker"))
    for i in range(1):
        logs = solver.step()
        if i % 100 == 0:
            print(i, logs, exploitability.exploitability(solver._game, solver))


if __name__ == '__main__':
   main()
