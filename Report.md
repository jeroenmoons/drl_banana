# Report

## Algorithm

Since the state space of this environment is continuous (agent velocity is a scalar, for example) we need to either 
discretize it or use a value function approximation method. I chose to go with DQN for this reason. I do suspect the 
problem is quite easily solvable using tiling as well though.

## Q network architecture

A simple fully connected network with two hidden layers sufficed to solve the environment within 500 episodes. (The 
extra convolutional layers used in the Deepmind paper are probably not really necessary since we're not dealing with 
images)

It looks like this:
* input of size 37 (state vector size)
* FC hidden layer of size 64
* FC hidden layer of size 64
* FC output layer of size 4 (action space size)

## Hyperparameters

I did some rudimentary manual grid search of the hyperparameter space and ended up with the following:

* alpha: 0.0005
* gamma: 0.99
* learning batch size: 100
* replay buffer size: 10^5
* epsilon: starts at 1, decays with factor .9999, bottoms out at 0.01
* tau: 0.001

## Results

### First solution

My first solution considers the environment solved when reaching an average score of 13 over 100 episodes.

The algorithm as outlined above reached a solution in about 400 iterations (well below the 1800 episodes the project 
description mentions). 

Training output:
```
Training agent.
Iteration 100 - avg score of 4.29 over last 100 episodes
Iteration 200 - avg score of 9.88 over last 100 episodes
Iteration 300 - avg score of 10.87 over last 100 episodes
Iteration 400 - avg score of 12.98 over last 100 episodes
Environment solved in 401 iterations with a score of 13.01
Training ended with an avg score of 13.01 over last 100 episodes
Max score: 24.0
```

Plotted scores show a steep initial learning phase followed by a slower but steady increase to the goal of 13+ average:
![scores](assets/first_solution_scores.png)
![average scores](assets/first_solution_avg_scores.png)

TODO: gif of trained agent.

### Second solution

I then tried to squeeze a bit more out of the algorithm, both by trying to reach a higher average score within 1800 
episodes and by playing with the parameters and network a bit more.

* TODO: pushing a bit further - describe results.


## Ideas for future work

1. A more structured and thorough search of the algorithm's parameters would most likely result in a more efficient 
solution. A **parallellized grid search** of the hyperparameter space would be interesting. 

2. An intriguing parameter is the **learning batch size**: a larger batch size seems to result in faster learning, but 
is this always the case? What is the limit? When does the necessary computation become too much for the added gain of 
learning faster?

3. This algorithm is a vanilla DQN implementation. This more than suffices to solve the problem at hand but implementing 
a few tweaks (or the full rainbow DQN) would almost certainly result in quicker convergence:
    * prioritized buffer replay to prefer learning from bad estimates
    * double DQN might learn more efficiently
    * noisy networks could be an interesting alternative for the epsilon parameter

4. Solving the problem from pixels instead of the ray traced state vectors, as suggested by the Udacity project 
description would be an interesting exercise, mainly because I suspect the agent would be more intelligent regarding
batches of bananas perceived in the distance (the agent using vector states only sees what's right in front of it).
I think it would learn to see a cluster of yellow bananas and decide it is worth the trip.