import torch
from poker_rnad import RNaD


def main():
    game_def = "leduc_poker(players=2)"
    print(game_def)

    rnad = RNaD(
        game_def=game_def,
        num_workers=8,
        num_threads=12,
        batch_size=768,
        max_queue_capacity=128,
        device=torch.device('cpu')
    )
    print("built rnad")
    rnad.run(1)


if __name__ == '__main__':
    main()
