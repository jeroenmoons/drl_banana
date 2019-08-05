# DRL Navigation - Banana Environment

This project applies Deep Reinforcement Learning on the Unity ML agents Banana environment.

More information on the algorithm, NN architecture and hyper-parameters can be found [in a separate report](Report.md).


## The Environment

TODO: describe environment, state space, action space.

### General

### State space

### Action space


## Getting started

### Installation

The project requires Python 3, PyTorch 0.4, the Unity ML API and the Unity ML environment which you can unzip in this 
project's bin directory (for Mac OS) or download from Udacity:
* Linux: <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip">click here</a>
* Mac OSX: <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip">click here</a>
* Windows (32-bit): <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip">click here</a>
* Windows (64-bit): <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip">click here</a>


To install the requirements, first create an Anaconda environment for the project using python 3.6:
```conda create --name env_name python=3.6 -y```

Activate the environment:
```conda activate env_name```

Then go to the project's python directory and install the requirements in your environment:
```pip install .```

Make sure the Unity environment is present in the `bin/` directory and the corresponding name has been set in the 
ENV_APP constant in `config.py`.


### Running the project

There are two ways to run the project. One trains an agent from scratch, the other shows you a pre-trained agent 
performing.


#### Training an agent

TODO: how to train a new agent from scratch, or resume training from a checkpoint.

#### Watching an agent

TODO: how to run a random or trained agent using command line switch.


---
_This project is part of my Udacity DRL Nanodegree_