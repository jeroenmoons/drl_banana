import config

import numpy as np
from unityagents import UnityEnvironment
from act import train
from agent.dqn import DqnAgent


def get_agent():
    """
    Builds the UnityAgent object to train.
    """

    # TODO: load checkpoint to continue learning

    # Create a new agent
    agent_params = {
        'device': config.PYTORCH_DEVICE,
        'alpha': 5e-4,
        'gamma': 0.99,
        'learn_batch_size': 100,
        'epsilon': 1.,
        'epsilon_decay': 0.9999,
        'epsilon_min': 0.01,
        'hidden_layer_sizes': (64, 64)
    }

    return DqnAgent(brain_name, state_size, action_size, agent_params)


if __name__ == '__main__':
    """
    This shows an agent performing a single episode.
    """
    print('training a new agent to master {}'.format(config.ENV_APP))

    # Create the Unity environment
    banana_env = UnityEnvironment(file_name=config.ENV_APP)

    # Select the brain (= Unity ML agent) to work with and examine action space
    brain_name = banana_env.brain_names[0]
    brain = banana_env.brains[brain_name]
    action_size = brain.vector_action_space_size

    # Examine state space
    env_info = banana_env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])

    agent = get_agent()
    print('Agent params: {}'.format(agent.get_params()))

    # Train the agent
    result = train(banana_env, agent)

    # Close the environment, no longer needed
    banana_env.close()

    print("Last 100 scores: {}".format(result[-100:]))
    print("Max score: {}".format(np.array(result).max()))
