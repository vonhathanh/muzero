import numpy as np
import torch

from args import Args
from utils import dict_to_cpu


class MLP(torch.nn.Module):

    def __init__(self, args: Args):
        super().__init__()
        self.action_space = args.action_space

        if torch.cuda.is_available() and args.train_on_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = "cpu"

        self.representation_net = mlp(np.array(args.observation_shape).prod(),
            [args.hidden_layers_size],
            args.encoding_size)

        self.dynamics_encoded_state_net = mlp(args.encoding_size + len(args.action_space),
                                              [args.hidden_layers_size],
                                              args.encoding_size)

        self.full_support_size = args.support_size * 2 + 1
        self.dynamics_reward_net = mlp(args.encoding_size, [args.hidden_layers_size], self.full_support_size)

        self.prediction_policy_net = mlp(args.encoding_size, [args.hidden_layers_size], len(args.action_space))

        self.prediction_value_net = mlp(args.encoding_size, [args.hidden_layers_size], self.full_support_size)

        if args.train_on_gpu:
            self.dynamics_reward_net.to(self.device)
            self.prediction_policy_net.to(self.device)
            self.prediction_value_net.to(self.device)

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_net(encoded_state)
        value = self.prediction_value_net(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_net(observation)
        return self.scale_encoded_state(encoded_state)

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = torch.nn.functional.one_hot(torch.tensor([action]).to(self.device), len(self.action_space))
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_state = self.dynamics_encoded_state_net(x)
        reward = self.dynamics_reward_net(next_state)

        return self.scale_encoded_state(next_state), reward

    @staticmethod
    def scale_encoded_state(s):
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = s.min(1, keepdim=True)[0]
        max_encoded_state = s.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (s - min_encoded_state) / scale_encoded_state

        return encoded_state_normalized

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.zeros((1, self.full_support_size), device=self.device)
        return value, reward, policy_logits, encoded_state

    def recurrent_inference(self, encoded_state, action):
        next_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_state)
        return value, reward, policy_logits, next_state

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


def mlp(
        input_size,
        layer_sizes,
        output_size,
        output_activation=torch.nn.Identity,
        activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)
