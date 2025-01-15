import tyro
import torch
import numpy as np
import multiprocessing as mp

from args import Args
from games.cartpole import Game
from model import MLP
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage


class Muzero:
    def __init__(self, args: Args):
        self.args = args
        self.shared_storage = SharedStorage()
        self.replay_buffer = ReplayBuffer(args)

        self.shared_storage.save_network(MLP(args))

    def train(self):
        for i in range(self.args.num_workers):
            self.launch_job(self.run_self_play, i + 1)
        # self.train_networks()

    def run_self_play(self, i):
        while True:
            model = self.shared_storage.get_latest_checkpoint()
            game = self.play_game(model)
            print(f"actor: {i} is playing game")
            self.replay_buffer.save_game(game)

    def play_game(self, model: MLP):
        game = Game(self.args)
        done = False

        while not done and len(game.history) < self.args.max_moves:
            root = Node(0)
            observation, _ = game.reset()
        #     self.expand_node(root, game.legal_actions(), model.initial_inference(observation))
        #     self.add_exploration_noise(root)
        #     self.run_mcts(root, game.history)
        #     action = self.select_action(len(game.history), root, model)
        #     observation, done = game.step(action)
        #     game.store_statistics(root)

        return game

    def launch_job(self, fn, *args):
        p = mp.Process(target=fn, args=args)
        p.start()


if __name__ == '__main__':
    mp.freeze_support()

    args = tyro.cli(Args)
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    muzero = Muzero(args)
    muzero.train()
