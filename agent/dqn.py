import numpy as np
import torch

from agent.base import UnityAgent
from agent.estimate.neural import FullyConnectedNetwork
from collections import namedtuple, deque


class DqnAgent(UnityAgent):
    """Chooses epsilon-greedy actions using a NN to estimate action values."""

    # Default params
    DEVICE_DEFAULT = 'cpu'  # pytorch device

    HIDDEN_LAYER_SIZES_DEFAULT = (50, 50)  # default q network hidden layer sizes

    REPLAY_BUFFER_SIZE = 100000  # max nr of experiences in memory

    EPSILON_DEFAULT = 1.  # starting value for epsilon
    EPSILON_DECAY_DEFAULT = .9999  # used to decay epsilon over time
    EPSILON_MIN_DEFAULT = .005  # minimum value for decayed epsilon

    LEARN_BATCH_SIZE = 50  # batch size to use when learning from memory

    def __init__(self, brain_name, state_size, action_size, params):
        super().__init__(brain_name, state_size, action_size, params)

        # pytorch device
        self.device = params.get('device', self.DEVICE_DEFAULT)

        # learning parameters
        self.epsilon = params.get('epsilon', self.EPSILON_DEFAULT)
        self.epsilon_decay = params.get('epsilon_decay', self.EPSILON_DECAY_DEFAULT)
        self.epsilon_min = params.get('epsilon_min', self.EPSILON_MIN_DEFAULT)

        self.learn_batch_size = params.get('learn_batch_size', self.LEARN_BATCH_SIZE)

        # memory
        self.memory_size = params.get('memory_size', self.REPLAY_BUFFER_SIZE)
        self.memory = ReplayBuffer(action_size, self.memory_size)

        # online and target Q-network models
        self.hidden_layer_sizes = params.get('hidden_layer_sizes', self.HIDDEN_LAYER_SIZES_DEFAULT)

        self.online_network = FullyConnectedNetwork(self.state_size, self.hidden_layer_sizes, self.action_size)
        self.target_network = FullyConnectedNetwork(self.state_size, self.hidden_layer_sizes, self.action_size)

    def select_action(self, state):
        """
        Selects an epsilon-greedy action from the action space, using the online_network to estimate action values.
        """

        # with probability epsilon, explore by choosing a random action
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)

        # else, use the online network to choose the action it currently estimates to be the best one
        state_tensor = torch.from_numpy(state).float()
        state_tensor = state_tensor.unsqueeze(0)  # wrap state in extra array so vector becomes a (single state) batch
        state_tensor = state_tensor.to(self.device)  # move the tensor to the configured device (cpu or cuda/gpu)

        action_values = self.online_network(state_tensor)
        best_action = torch.argmax(action_values.squeeze()).numpy().item(0)  # pick action with highest value

        print('action_values', action_values)
        print('action_values numpy', action_values.squeeze().detach().numpy())
        print('best action: ', best_action)

        return best_action

    def step(self, state, action, result):
        next_state = result.vector_observations[0]
        reward = result.rewards[0]
        done = result.local_done[0]

        self.memory.add(state, action, reward, next_state, done)

        # TODO:
        #  - Learn by sampling from buffer and training live network every X steps
        #  - Copy online network to target network every Y steps

        print('experiences in memory: ', len(self.memory))

        return reward, done

    def get_params(self):
        return {
            'device': self.device,
            'memory_size': self.memory_size,
            'learn_batch_size': self.learn_batch_size,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'online_network': self.online_network,
            'target_network': self.target_network
        }


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size):
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
