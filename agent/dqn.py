import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent.base import UnityAgent
from agent.estimate.neural import FullyConnectedNetwork
from collections import namedtuple, deque


class DqnAgent(UnityAgent):
    """Chooses epsilon-greedy actions using a NN to estimate action values."""

    # Default params
    DEVICE_DEFAULT = 'cpu'  # pytorch device

    HIDDEN_LAYER_SIZES_DEFAULT = (50, 50)  # default q network hidden layer sizes

    REPLAY_BUFFER_SIZE = 100000  # max nr of experiences in memory

    ALPHA_DEFAULT = .1  # default learning rate

    GAMMA_DEFAULT = .9  # default reward discount factor

    EPSILON_DEFAULT = 1.  # starting value for epsilon
    EPSILON_DECAY_DEFAULT = .9999  # used to decay epsilon over time
    EPSILON_MIN_DEFAULT = .005  # minimum value for decayed epsilon

    LEARN_BATCH_SIZE = 50  # batch size to use when learning from memory

    def __init__(self, brain_name, state_size, action_size, params):
        super().__init__(brain_name, state_size, action_size, params)

        # pytorch device
        self.device = params.get('device', self.DEVICE_DEFAULT)

        # learning parameters
        self.alpha = params.get('alpha', self.ALPHA_DEFAULT)
        self.gamma = params.get('gamma', self.GAMMA_DEFAULT)

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

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.alpha)

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

        self.online_network.eval()  # switch to evaluation mode for more efficient evaluation of the state tensor
        with torch.no_grad():
            action_values = self.online_network(state_tensor)
        self.online_network.train()  # and back to training mode

        best_action = torch.argmax(action_values.squeeze()).numpy().item(0)  # pick action with highest Q value

        print('action_values', action_values)
        print('action_values numpy', action_values.squeeze().detach().numpy())
        print('best action: ', best_action)

        return best_action

    def step(self, state, action, result):
        next_state = result.vector_observations[0]
        reward = result.rewards[0]
        done = result.local_done[0]

        self.memory.add(state, action, reward, next_state, done)
        print('experiences in memory: ', len(self.memory))

        experiences = self.memory.sample(self.learn_batch_size)

        self.learn(experiences, self.gamma)

        return reward, done

    def learn(self, experiences, gamma):
        """Performs gradient descent of the local network on the batch of experiences."""

        # create pytorch tensors from the experiences
        states = torch.tensor(np.array([e.state for e in experiences])).float().to(self.device)
        actions = torch.tensor(np.array([[e.action] for e in experiences])).long().to(self.device)
        rewards = torch.tensor(np.array([[e.reward] for e in experiences])).float().to(self.device)
        next_states = torch.tensor(np.array([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.tensor(np.array([[1] if e.done else [0] for e in experiences]).astype(np.uint8)).float().to(self.device)

        # get q values for next states from target_network
        next_state_values = self.target_network(next_states)

        # get the maximum q value for each of the next_states
        best_next_action_values = next_state_values.detach().max(1)[0].unsqueeze(1)

        # calculate q values for current states: gamma * reward + target_value_of_next_state
        # multiplication by (1 - dones) sets next state value to 0 if it is end of episode.
        targets = rewards + (gamma * best_next_action_values * (1 - dones))

        # calculate expected q values for current state by evaluation through online_network
        expecteds = self.online_network(states).gather(1, actions)

        # calculate error between the two (= loss)
        loss = F.mse_loss(expecteds, targets)

        # minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TODO: Perform soft update of the target network parameters from online network (governed by TAU hyper param)
        pass

    def get_params(self):
        return {
            'device': self.device,
            'memory_size': self.memory_size,
            'learn_batch_size': self.learn_batch_size,
            'alpha': self.alpha,
            'gamma': self.gamma,
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

    def sample(self, sample_size):
        """Select a random batch of experiences from the buffer."""
        memory_size = len(self.buffer)
        sample_size = sample_size if sample_size < memory_size else memory_size

        return random.sample(self.buffer, k=sample_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
