"""
Contains constants configuring the project and training process.
"""
import torch

# Unity built environment app
ENV_APP = 'bin/Banana.app'

# Device to run pytorch calculations on
PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training loop control
MAX_ITERATIONS = 10  # maximum number of episodes to train on
MAX_EPISODE_STEPS = 5000  # cap the maximum steps within an episode

# Average score over 100 episodes to be reached to solve the environment
SOLVED_SCORE = 13
