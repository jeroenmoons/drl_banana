import numpy as np
from agent.base import UnityAgent


class DqnAgent(UnityAgent):
    # Default params
    EPSILON_DEFAULT = 1.
    EPSILON_DECAY_DEFAULT = .9999
    EPSILON_MIN_DEFAULT = .005

    """
    Chooses actions using a NN to estimate action values.
    """
    def __init__(self, brain_name, state_size, action_size, params):
        super().__init__(brain_name, state_size, action_size)

        self.epsilon = params.get('epsilon', self.EPSILON_DEFAULT)
        self.epsilon_decay = params.get('epsilon', self.EPSILON_DECAY_DEFAULT)
        self.epsilon_min = params.get('epsilon', self.EPSILON_MIN_DEFAULT)

    def select_action(self, state):
        return np.random.choice(self.action_size)  # TODO: implement DQN epsilon-greedy action selection

    def step(self, env_info):
        # TODO: agent-specific step handling
        # Add experience to buffer
        # Learn by sampling from buffer and training live network

        pass
