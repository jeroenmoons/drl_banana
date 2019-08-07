import config

import numpy as np
import matplotlib.pyplot as plt


def train(env, agent):
    """
    Performs the main training loop.
    """

    scores = []
    scores_avg = []
    iterations = 0
    solved = False  # TODO

    while iterations < config.MAX_ITERATIONS and not solved:
        iterations += 1

        done = False
        score = 0
        env_info = env.reset(train_mode=True)[agent.brain_name]  # reset the environment

        episode_steps = 0
        while not done and episode_steps < config.MAX_EPISODE_STEPS:
            episode_steps += 1

            state = env_info.vector_observations[0]
            action = agent.select_action(state)  # choose an action
            env_info = env.step(action)[agent.brain_name]  # execute that action
            reward, done = agent.step(state, action, env_info)  # give the agent the chance to learn from the results

            score += reward  # update score with the reward

        if iterations % 100 == 0:
            mean = np.mean(scores[-100:])
            print('Iteration {} - scored {} on average in the last 100 episodes'.format(iterations, mean))

        # Keep track of scores for plotting a running average and whether or not the env is solved
        scores.append(score)
        scores_avg.append(np.mean(scores[-100:]))

    # plot all scores
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # plot average scores
    plt.plot(np.arange(len(scores_avg)), scores_avg)
    plt.ylabel('Avg Score over last 100 eps')
    plt.xlabel('Episode #')
    plt.show()

    return scores


def run_episode(env, agent):
    """
    Shows a UnityAgent perform a single episode in the specified Unity ML environment.
    """
    done = False
    score = 0  # initialize score

    env_info = env.reset(train_mode=False)[agent.brain_name]  # reset the environment

    while not done:
        state = env_info.vector_observations[0]  # observe the current env state
        action = agent.select_action(state)  # choose an action
        env_info = env.step(action)[agent.brain_name]  # execute that action
        score += env_info.rewards[0]  # update episode score with the reward obtained from the action
        done = env_info.local_done[0]  # check if episode has finished

    return score
