import numpy as np
from agent.base import UnityAgent


class DqnAgent(UnityAgent):
    """
    Chooses actions using a NN to estimate action values.
    """
    def select_action(self, state):
        return np.random.choice(self.action_size)  # TODO: implement DQN epsilon-greedy action selection
