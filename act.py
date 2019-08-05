def run_episode(env, agent):
    """
    Shows a UnityAgent perform a single episode in the specified Unity ML environment.
    """
    score = 0  # initialize score

    env_info = env.reset(train_mode=False)[agent.brain_name]  # reset the environment

    while True:
        state = env_info.vector_observations[0]
        action = agent.select_action(state)  # pick random action
        env_info = env.step(action)[agent.brain_name]  # execute that action
        done = env_info.local_done[0]  # check if episode has finished
        score += env_info.rewards[0]  # update score with the reward

        if done:  # stop if episode has finished
            break

    return score
