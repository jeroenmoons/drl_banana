# Report

## Algorithm

Since the state space of this environment is continuous (agent velocity is a scalar, for example) we need to either 
discretize it or use a value function approximation method. I chose to go with DQN for this reason. I do suspect the 
problem is quite easily solvable using tiling as well though.

## Q network architecture

A simple fully connected network with two hidden layers sufficed to solve the environment within 500 episodes.

It looks like this:
* input layer of size 37 (state vector size)
* hidden layer of size 64
* hidden layer of size 64
* output layer of size 4 (action space size)

## Hyperparameters

I did some rudimentary manual grid search of the hyperparameter space and ended up with the following:

* alpha: 0.0005
* gamma: 0.99
* learning batch size: 100
* replay buffer size: 10^5
* epsilon: starts at 1, decays with factor .9999, bottoms out at 0.01
* tau: 0.001

## Results

# TODO
* solved how fast? avg score?
* score evolution?
* pushing a bit further
* gif of trained agent?

## Ideas for future work

A more structured and thorough search of the algorithm's parameters would most likely result in a more efficient 
solution. A parallellized grid search of the hyperparameter space would be interesting.

This algorithm is a vanilla DQN implementation. This more than suffices to solve the problem at hand but implementing a 
few tweaks would probably result in quicker convergence:
* prioritized buffer replay to prefer learning from bad estimates
* double DQN might learn faster
* noisy networks could be an interesting alternative for the epsilon parameter

Solving the environment from pixels, as suggested by the Udacity project description would be an interesting exercise.