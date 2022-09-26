# Obstacle avoidance with DeepQ Reinforcement Learning
### Inspired by Retro Atari Game "River Raid"

The model's task is to get as far as possible, whitout colliding with green walls.

The only information that model gets is the current state (80x80 RGB map) and list of possible actions (LEFT, RIGHT, None).

Based on DeepQ Reinforcement Learning technique - model tries to learn Q-function by taking actions and evaluating them in the future.

Resources:
https://keras.io/examples/rl/deep_q_network_breakout/
http://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L5.pdf

#### Results after 4000 runs:

![](https://github.com/jjp241/deepq_river_raid/blob/master/demo/4000_runs.gif)

## How to run it yourself?

Clone repo and install requirements specified in `requirements.txt`. Then `python3 run.py` will run current model (4000 runs) and display its run.

## How do I train it?

Specify your hyperparameters in `models/deepq.py` and uncomment training section in `run.py`. Then `python3 run.py` will do the job :)
