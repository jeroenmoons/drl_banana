import config

from agent.dqn import DqnAgent
from agent.random import RandomAgent


class AgentFactory:
    """Responsible for creating UnityAgent instances."""
    def create_agent(self, agent_type, brain_name, state_size, action_size):
        if agent_type == 'random':
            return self.create_random_agent(brain_name, state_size, action_size)
        elif agent_type == 'dqn_new':
            return self.create_dqn_agent(brain_name, state_size, action_size)
        elif agent_type == 'dqn_checkpoint':
            return self.create_dqn_agent(brain_name, state_size, action_size, checkpoint='dqn_agent_checkpoint.pth')
        elif agent_type == 'dqn_solved':
            return self.create_dqn_agent(brain_name, state_size, action_size, checkpoint='dqn_agent_solved.pth')
        elif agent_type == 'dqn_pretrained':
            return self.create_pretrained_dqn_agent(brain_name, state_size, action_size)
        else:
            raise ValueError('Unknown agent type {}'.format(agent_type))

    def create_random_agent(self, brain_name, state_size, action_size):
        """Creates a random agent."""
        return RandomAgent(brain_name, state_size, action_size, {})

    def create_dqn_agent(self, brain_name, state_size, action_size, checkpoint=None):
        """Creates a DQN agent with the params specified below, optionally loads network weights from checkpoint."""
        agent_params = {
            'device': config.PYTORCH_DEVICE,
            'alpha': 1e-4,
            'gamma': 0.99,
            'learn_batch_size': 100,
            'epsilon': 1.,
            'epsilon_decay': 0.9999,
            'epsilon_min': 0.01,
            'hidden_layer_sizes': (25, 25)
        }

        agent = DqnAgent(brain_name, state_size, action_size, agent_params)

        if checkpoint is not None:
            agent.load_checkpoint('saved_models/{}'.format(checkpoint))

        return agent

    def create_pretrained_dqn_agent(self, brain_name, state_size, action_size, checkpoint=None):
        """Creates a pre-trained DQN agent."""
        agent_params = {
            'device': config.PYTORCH_DEVICE,
            'alpha': 1e-4,
            'gamma': 0.99,
            'learn_batch_size': 100,
            'epsilon': 1.,
            'epsilon_decay': 0.9999,
            'epsilon_min': 0.01,
            'hidden_layer_sizes': (25, 25)
        }

        agent = DqnAgent(brain_name, state_size, action_size, agent_params)
        agent.load_checkpoint('saved_models/dqn_agent_pretrained.pth')

        return agent
