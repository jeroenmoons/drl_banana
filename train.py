import config

import sys
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from agent.factory import AgentFactory


def train(env, agent):
    """
    Performs the main training loop.
    """

    max_score = 0
    scores = []
    scores_avg = []
    iterations = 0
    solved = False

    print('Training agent.')

    while iterations < config.MAX_ITERATIONS and not solved:
        # show a progress indication
        print('\repisode {}, max score so far is {}'.format(iterations, max_score), end='')
        sys.stdout.flush()

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

        scores.append(score)  # keep track of the episode score
        avg_score = np.mean(scores[-100:])  # calculate average score over the last 100 episodes
        scores_avg.append(avg_score)  # keep track of the average score
        max_score = score if max_score < score else max_score  # keep track of max score so far

        # print periodic progress report
        if iterations % 100 == 0:
            print('\rIteration {} - avg score of {} over last 100 episodes'.format(iterations, avg_score))
            agent.save_checkpoint(name='checkpoint')

        # if the environment is solved, stop training
        if not solved and avg_score > config.SOLVED_SCORE:
            print('\rEnvironment solved in {} iterations with a score of {}'.format(iterations, avg_score))
            solved = True
            agent.save_checkpoint(name='solved')

    print('Training ended with an avg score of {} over last 100 episodes'.format(scores_avg[-1]))
    plot_scores(scores, scores_avg)

    return scores


def plot_scores(scores, scores_avg):
    """Creates plots of score track record."""

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
    initial_env_info = banana_env.reset(train_mode=True)[brain_name]
    state_size = len(initial_env_info.vector_observations[0])

    # Create a new DQN agent
    agent_factory = AgentFactory()
    an_agent = agent_factory.create_agent('dqn_new', brain_name, state_size, action_size)
    an_agent.training = True  # True is default, but just in case
    print('Agent params: {}'.format(an_agent.get_params()))

    # Train the agent
    result = train(banana_env, an_agent)

    # Close the environment, no longer needed
    banana_env.close()

    print("Max score: {}".format(np.array(result).max()))
