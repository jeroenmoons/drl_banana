import config

from unityagents import UnityEnvironment


def train(env, brain_name):
    """
    Performs the main DQN training loop.
    """
    score = 0  # initialize score

    return score


if __name__ == '__main__':
    """
    This runs a randomly acting agent, completely ignoring the state and picking a random action at each time step.
    """
    print('training a new agent to master {}'.format(config.ENV_APP))

    banana_env = UnityEnvironment(file_name=config.ENV_APP)  # Create the Unity environment

    # TODO: create a new DQN agent

    # Train the agent
    result = train(banana_env,
                   banana_env.brain_names[0])

    banana_env.close()  # Close the environment, no longer needed

    print("Score: {}".format(result))
