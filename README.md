# DRL Navigation - Banana Environment

This project applies Deep Reinforcement Learning on the 
[Unity ML agents](https://github.com/Unity-Technologies/ml-agents) Banana environment.

More information on the algorithm, NN architecture and hyper-parameters can be found [in this report](Report.md).


## The Environment

The environment consists of a closed room containing yellow and blue bananas. 

![env](assets/banana.gif)

The goal is to find yellow bananas and avoid blue bananas. It is an episodic environment, the agent gets a fixed number 
of steps in which to maximise its reward.

The environment is considered to be solved if an agent gets an average reward of at least 13 over 100 episodes.

### Rewards

A reward of +1 is earned by catching a yellow banana. 
A penalty of -1 is given for catching a blue banana. 
Reward at all other times is 0.

This embodies the goal of catching as many yellow bananas as possible and avoiding the blue ones.

### State space

The state space is continuous. It consists of vectors of size 37, specifying the agent's velocity and a ray-traced 
representation of the agent's local field of vision. It specifies presence of any objects under a number of fixed angles 
in front of the agent.

### Action space

The action space is discrete and consists of four options:
* go forward (0)
* go backward (1)
* go left (2)
* go right (3)


## Getting started
### Installation

The project requires Python 3, PyTorch 0.4, the Unity ML API and the Unity ML environment which you can unzip in this 
project's bin directory (for Mac OS) or download from Udacity:
* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [Mac OS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


To install the requirements, first create an [Anaconda environment](https://www.anaconda.com/distribution/) (or another 
virtual env of your choice) for the project using python 3.6: ```conda create --name env_name python=3.6 -y```

Activate the environment:
```conda activate env_name```

Then go to the project's python directory and install the requirements in your environment:
```pip install .```

Make sure the Unity environment is present in the `bin/` directory and the corresponding name has been set in the 
ENV_APP constant in `config.py`.


### Running the project

The project is run from the command line. There are two entry points. One trains an agent from scratch, the other shows 
you an agent performing a single episode.

#### 1. `train.py` - Training an agent

To train a new agent, run the `train.py` script. It currently only supports training a DQN agent so no command line 
arguments are necessary. (Training parameters are set in config.py and the agent/factory.py module.)

When the environment is solved, the training script saves a checkpoint in the saved_models directory. During training, a 
checkpoint is saved every 100 iterations as wel. Both can be loaded with the watch script (see next point).

#### 2. `watch.py` - Watching an agent

To watch an agent perform a single episode, run the `watch.py` script and specify which agent you would like to see using
the `--agent` option. Available choices are:

* `random`: shows a perfectly stupid agent.
* `dqn_pretrained`: shows a pre-trained agent.
* `dqn_checkpoint`: shows the last saved checkpoint reached during training.
* `dqn_solved`: shows the last solution reached by the training script.

Example: ```python watch.py --agent=dqn_pretrained```

---
_This project is part of my Udacity DRL Nanodegree_