import config


def train(env, agent):
    """
    Performs the main training loop.
    """

    # TODO
    #  - Decay epsilon (should be an agent internal)
    #  - Run 'regular' episode every X iterations (train_mode=False) so we can see the agent.
    #  - Action selection histogram?

    scores = []
    iterations = 0
    solved = False  # TODO

    while iterations < config.MAX_ITERATIONS and not solved:
        iterations += 1
        print('iteration {}'.format(iterations))

        done = False
        score = 0
        env_info = env.reset(train_mode=True)[agent.brain_name]  # reset the environment

        episode_steps = 0
        while not done and episode_steps < config.MAX_EPISODE_STEPS:
            episode_steps += 1

            state = env_info.vector_observations[0]
            action = agent.select_action(state)  # choose an action
            print('chose action {}'.format(action))

            env_info = env.step(action)[agent.brain_name]  # execute that action
            reward, done = agent.step(state, action, env_info)  # give the agent the chance to do something with the results

            score += reward  # update score with the reward

        print('scored {}'.format(score))

        # Keep track of scores for plotting a running average and whether or not the env is solved
        scores.append(score)

    # TODO: periodic feedback
    # TODO: plot score average evolution
    # TODO: keep track of maximum score, plot

    return scores


def run_episode(env, agent):
    """
    Shows a UnityAgent perform a single episode in the specified Unity ML environment.
    """
    done = False
    score = 0  # initialize score

    env_info = env.reset(train_mode=False)[agent.brain_name]  # reset the environment

    while not done:
        state = env_info.vector_observations[0]
        action = agent.select_action(state)  # choose an action
        env_info = env.step(action)[agent.brain_name]  # execute that action
        done = env_info.local_done[0]  # check if episode has finished
        score += env_info.rewards[0]  # update score with the reward

    return score
