import config

from unityagents import UnityEnvironment
from act import train
from agent.dqn import DqnAgent


if __name__ == '__main__':
    """
    This runs a randomly acting agent, completely ignoring the state and picking a random action at each time step.
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

    # Create a new agent - TODO: command line agent type selection, load checkpoint
    agent = DqnAgent(brain_name, state_size, action_size)

    # Train the agent
    result = train(banana_env, agent)

    banana_env.close()  # Close the environment, no longer needed

    print("Score: {}".format(result))
