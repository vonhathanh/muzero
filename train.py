import tyro
import torch
import numpy as np

from args import Args
from games.cartpole import Game
from model import MLP
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage


class Muzero:
    def __init__(self, args: Args):
        self.args = args
        self.game = Game(args.seed)
        self.model = MLP(args)
        self.shared_storage = SharedStorage()
        self.replay_buffer = ReplayBuffer(args)

    def train(self):
        for i in range(self.args.num_workers):
            self.launch_job(self.play_game, i+1)
        self.train_networks()

    def play_game(self, i):
        print(f"actor: {i} is playing game")

    def launch_job(self, fn, *args):
        fn(*args)


if __name__ == '__main__':
    args = tyro.cli(Args)
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    muzero = Muzero(args)
    muzero.train()
