from agent.random import RandomAgent


class AgentFactory:
    """Responsible for creating UnityAgent instances."""
    def create_agent(self, agent_type, brain_name, state_size, action_size):
        if agent_type == 'random':
            return self.create_random_agent(brain_name, state_size, action_size)
        else:
            raise ValueError('Unknown agent type {}'.format(agent_type))

    def create_random_agent(self, brain_name, state_size, action_size):
        return RandomAgent(brain_name, state_size, action_size, {})
