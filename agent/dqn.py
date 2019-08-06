import numpy as np
from agent.base import UnityAgent
from torch import nn


class DqnAgent(UnityAgent):
    """
    Chooses actions using a NN to estimate action values.
    """

    # Default params
    EPSILON_DEFAULT = 1.  # starting value for epsilon
    EPSILON_DECAY_DEFAULT = .9999  # used to decay epsilon over time
    EPSILON_MIN_DEFAULT = .005  # minimum value for decayed epsilon

    HIDDEN_LAYER_SIZES_DEFAULT = (50, 50)  # default q network hidden layer sizes

    def __init__(self, brain_name, state_size, action_size, params):
        super().__init__(brain_name, state_size, action_size, params)

        # learning parameters
        self.epsilon = params.get('epsilon', self.EPSILON_DEFAULT)
        self.epsilon_decay = params.get('epsilon_decay', self.EPSILON_DECAY_DEFAULT)
        self.epsilon_min = params.get('epsilon_min', self.EPSILON_MIN_DEFAULT)

        # online and target Q-network models
        self.hidden_layer_sizes = params.get('hidden_layer_sizes', self.HIDDEN_LAYER_SIZES_DEFAULT)

        self.online_network = Network(self.state_size, self.hidden_layer_sizes, self.action_size)
        self.target_network = Network(self.state_size, self.hidden_layer_sizes, self.action_size)

    def select_action(self, state):
        return np.random.choice(self.action_size)  # TODO: implement DQN epsilon-greedy action selection

    def step(self, env_info):
        # TODO: agent-specific step handling
        # Add experience to buffer
        # Learn by sampling from buffer and training live network

        pass

    def get_params(self):
        return {
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'online_network': self.online_network,
            'target_network': self.target_network
        }


class Network(nn.Module):
    """Builds a neural network to estimate action (Q) values for a state"""

    def __init__(self, input_size, hidden_sizes, output_size):
        super(Network, self).__init__()

        layers = []

        # add input and hidden layers
        size_in = input_size
        for size_out in hidden_sizes:
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.ReLU())

            size_in = size_out  # layer output size is size_in for the next layer

        layers.append(nn.Linear(size_in, output_size))  # add the output layer

        self.model = nn.Sequential(*layers)  # create a model from the layers

    def forward(self, x):
        return self.model.forward(x)  # delegate to the underlying sequential model
