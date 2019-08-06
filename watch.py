import config

from unityagents import UnityEnvironment
from agent.random import RandomAgent
from act import run_episode


if __name__ == '__main__':
    """
    Watch an agent during a single episode.
    """
    # Create the Unity environment
    banana_env = UnityEnvironment(file_name=config.ENV_APP)

    # Select the brain (= Unity ML agent) to work with and examine action space
    brain_name = banana_env.brain_names[0]
    brain = banana_env.brains[brain_name]
    action_size = brain.vector_action_space_size

    # Examine state space
    env_info = banana_env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])

    # Create a random agent - TODO: switch between random and trained using command line argument
    agent = RandomAgent(brain_name, state_size, action_size, {})

    # Run the agent inside the Banana environment
    result = run_episode(banana_env, agent)

    banana_env.close()

    print("Score: {}".format(result))
