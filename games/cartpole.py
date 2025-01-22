import gymnasium as gym
import torch

from action_history import ActionHistory
from games.abstract_game import AbstractGame


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, action_space, seed=0):
        self.env = gym.make("CartPole-v1")
        self.action_space = action_space
        self.seed = seed
        self.history = []

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, terminated, truncated, _ = self.env.step(action)
        return torch.tensor([observation]), reward, terminated or truncated

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(2))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation, info = self.env.reset(seed=self.seed)
        return torch.tensor([observation])

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Push cart to the left",
            1: "Push cart to the right",
        }
        return f"{action_number}. {actions[action_number]}"

    def action_history(self):
        return ActionHistory(self.history, self.action_space)