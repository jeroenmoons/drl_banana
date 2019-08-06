class UnityAgent:
    """
    Abstract Unity ML Reinforcement Learning agent base class.
    """
    def __init__(self, brain_name, state_size, action_size, params):
        self.brain_name = brain_name
        self.state_size = state_size
        self.action_size = action_size
        self.params = params

    def select_action(self, state):
        pass  # Should be implemented by child classes

    def step(self, env_info):
        pass  # Should be implemented by child classes

    def get_params(self):
        return self.params
