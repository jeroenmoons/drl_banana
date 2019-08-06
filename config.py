"""
Contains constants configuring the project and training process.
"""
import torch

# Unity built environment app
ENV_APP = 'bin/Banana.app'

# Device to run pytorch calculations on
PYTORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Average score over 100 episodes to be reached to solve the environment
SOLVED_SCORE = 13
