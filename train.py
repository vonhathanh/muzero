import math

import tyro
import torch
import numpy as np
import multiprocessing as mp

from action_history import ActionHistory
from args import Args
from games.cartpole import Game
from model import MLP
from node import Node
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
        game = Game(self.args.seed)
        done = False

        while not done and len(game.history) < self.args.max_moves:
            root = Node(0)
            observation = game.reset()
            self.expand_node(root, game.legal_actions(), model.initial_inference(observation))
            self.add_exploration_noise(root)
            self.run_mcts(root, game.history, model)
        #     action = self.select_action(len(game.history), root, model)
        #     observation, done = game.step(action)
        #     game.store_statistics(root)

        return game

    def run_mcts(self, root: Node, history: ActionHistory, model: MLP):
        pass

    def expand_node(self, node: Node, legal_actions: list | tuple, model_output: tuple):
        value, reward, policy_logits, encoded_state = model_output

        node.hidden_state = encoded_state
        node.reward = reward

        policy = {a: math.exp(policy_logits[a]) for a in legal_actions}
        policy_sum = sum(policy.values())

        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)

    def add_exploration_noise(self, node: Node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.args.root_dirichlet_alpha] * len(actions))
        frac = self.args.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def launch_job(self, fn, *args):
        fn(*args)
        # p = mp.Process(target=fn, args=args)
        # p.start()


if __name__ == '__main__':
    mp.freeze_support()

    args = tyro.cli(Args)
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    muzero = Muzero(args)
    muzero.train()
