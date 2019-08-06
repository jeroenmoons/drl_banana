import config


def train(env, agent):
    """
    Performs the main training loop.
    """
    scores = []

    # TODO
    #  - Stop after MAX_ITERATIONS or when solved (reached avg score of SOLVED_SCORE)
    #  - Decay epsilon (should be an agent internal)
    #  - Run 'regular' episode every X iterations (train_mode=False) so we can see the agent.

    done = False
    score = 0
    env_info = env.reset(train_mode=True)[agent.brain_name]  # reset the environment

    # For now, single step for easier debugging.
    # while not max_iterations or solved:  # TODO
    # while not done and episode_steps < max:  # TODO
    state = env_info.vector_observations[0]
    action = agent.select_action(state)  # choose an action
    print('chose action {}'.format(action))

    env_info = env.step(action)[agent.brain_name]  # execute that action
    reward, done = agent.step(state, action, env_info)  # give the agent the chance to do something with the results

    score += reward  # update score with the reward

    # Keep track of scores for plotting a running average and whether or not the env is solved
    scores.append(score)

    # TODO: periodic feedback
    # TODO: plot score average evolution

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
