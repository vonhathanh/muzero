from dataclasses import dataclass

import torch
import pathlib
import datetime


@dataclass
class Args:
    seed: int = 0  # Seed for numpy, torch and the game

    ### Game
    # Dimensions of the game observation, must be 3D (channel, height, width).
    # For a 1D array, please reshape it to (1, 1, length of array)
    observation_shape: tuple = (1, 1, 4)
    action_space: tuple = tuple(range(2))  # Fixed list of all possible actions. You should only edit the length
    players: tuple = tuple(range(1))  # List of players. You should only edit the length
    # Number of previous observations and previous actions to add to the current observation
    stacked_observations: int = 0

    # Evaluate
    muzero_player: int = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)

    ### Self-Play
    num_workers: int = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    selfplay_on_gpu: bool = False
    max_moves: int = 500  # Maximum number of moves if game is not finished before
    num_simulations: int = 50  # Number of future moves self-simulated
    discount: float = 0.997  # Chronological discount of the reward
    # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0
    # (ie selecting the best action). If -1, visit_softmax_temperature_fn is used every time
    temperature_threshold: int = -1

    # Root prior exploration noise
    root_dirichlet_alpha: float = 0.25
    root_exploration_fraction: float = 0.25

    # UCB formula
    pb_c_base: float = 19652
    pb_c_init: float = 1.25

    ### Network
    network: str = "fullyconnected"  # "resnet" / "fullyconnected"
    # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to
    # support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
    support_size: int = 10

    # Fully Connected Network
    encoding_size: int = 8
    hidden_layers_size: int = 16

    ### Training
    results_path: str = pathlib.Path(__file__).resolve().cwd() / "results" / datetime.datetime.now().strftime(
        "%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
    training_steps: int = 10000  # Total number of training steps (ie weights update according to a batch)
    batch_size: int = 128  # Number of parts of games to train on at each training step
    checkpoint_interval: int = 10  # Number of training steps before using the model for self-playing
    # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25
    value_loss_weight: float = 1.0
    train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

    optimizer: str = "Adam"  # "Adam" or "SGD". Paper uses SGD
    weight_decay: float = 1e-4  # L2 weights regularization
    momentum: float = 0.9  # Used only if optimizer is SGD

    # Exponential learning rate schedule
    lr_init: float = 0.02  # Initial learning rate
    lr_decay_rate: float = 0.8  # Set it to 1 to use a constant learning rate
    lr_decay_steps: int = 1000

    ### Replay Buffer
    replay_buffer_size: int = 500  # Number of self-play games to keep in the replay buffer
    num_unroll_steps: int = 10  # Number of game moves to keep for every batch element
    td_steps: int = 50  # Number of steps in the future to take into account for calculating the target value
    PER: bool = True  # Prioritized Replay, select in priority the elements in the replay buffer
    PER_alpha: float = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

    ### Adjust the self play / training ratio to avoid over/underfitting
    self_play_delay: float = 0  # Number of seconds to wait after each played game
    training_delay: float = 0  # Number of seconds to wait after each training step
    # Desired training steps per self played step ratio. Equivalent to a synchronous version,
    # training can take much longer. Set it to None to disable it
    training_steps_per_selfplay_step_ratio: float = 1.5
