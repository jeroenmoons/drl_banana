import config

import numpy as np
from unityagents import UnityEnvironment


def run(env, brain_name):
    print('Viewing a pre-trained agent in {}'.format(config.ENV_APP))

    action_size = env.brains[brain_name].vector_action_space_size

    env.reset(train_mode=False)[brain_name]  # reset the environment
    score = 0  # initialize score

    while True:
        action = np.random.randint(action_size)  # pick random action
        env_info = env.step(action)[brain_name]  # execute that action
        done = env_info.local_done[0]  # check if episode has finished
        score += env_info.rewards[0]  # update score with the reward

        if done:  # stop if episode has finished
            break

    return score


if __name__ == '__main__':
    '''
    This runs a randomly acting agent, completely ignoring the state and picking a random action at each time step.
    '''
    banana_env = UnityEnvironment(file_name=config.ENV_APP)
    result = run(banana_env, banana_env.brain_names[0])
    banana_env.close()

    print("Score: {}".format(result))
