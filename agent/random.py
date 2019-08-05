import numpy as np
from agent.base import UnityAgent


class RandomAgent(UnityAgent):
    """
    Randomly acting agent. Total disregard for environment state.
    """
    def select_action(self, state, epsilon=0.):
        return np.random.choice(self.action_size)
