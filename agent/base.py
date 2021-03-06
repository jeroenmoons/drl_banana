class UnityAgent:
    """Abstract Unity ML Reinforcement Learning agent base class."""

    def __init__(self, brain_name, state_size, action_size, params):
        self.brain_name = brain_name
        self.state_size = state_size
        self.action_size = action_size
        self.params = params
        self.training = True

    def select_action(self, state):
        pass  # Should be implemented by child classes

    def step(self, state, action, result):
        pass  # Should be implemented by child classes

    def save_checkpoint(self):
        """Save agent state for later retrieval."""
        pass  # Should be implemented by child classes

    def load_checkpoint(self, checkpoint):
        """Load agent state from checkpoint."""
        pass  # Should be implemented by child classes

    def get_params(self):
        return self.params
