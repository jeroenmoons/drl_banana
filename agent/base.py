class UnityAgent:
    """
    Abstract Unity ML Reinforcement Learning agent base class.
    """
    def __init__(self, brain_name, state_size, action_size):
        self.brain_name = brain_name
        self.state_size = state_size
        self.action_size = action_size

    def select_action(self, state):
        pass  # Should be implemented by child classes
