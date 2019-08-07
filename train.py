import config

import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from agent.dqn import DqnAgent


def get_agent(brain_name, state_size, action_size):
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

        scores.append(score)

        avg_score = np.mean(scores[-100:])

        if iterations % 100 == 0:
            print('Iteration {} - avg score of {} over last 100 episodes'.format(iterations, avg_score))

            # if the environment is solved, stop training
            if not solved and avg_score > config.SOLVED_SCORE:
                print('Environment solved with a score of {}'.format(avg_score))
                solved = True

        scores_avg.append(avg_score)

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

    an_agent = get_agent(brain_name, state_size, action_size)
    print('Agent params: {}'.format(an_agent.get_params()))

    # Train the agent
    result = train(banana_env, an_agent)

    # Close the environment, no longer needed
    banana_env.close()

    print("Last 100 scores: {}".format(result[-100:]))
    print("Max score: {}".format(np.array(result).max()))
