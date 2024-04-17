import time

import numpy as np
import torch
from torch import FloatTensor

import pettingzoo.mpe.simpla_pathfinding.simple_pathfinding as mpe

# import pettingzoo.mpe.simple_v3 as mpe

def random_act(var = 0.5):
    action = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    noise = np.random.randn(2) * var * 0.5

    if noise[0] < 0:
        action[1] = abs(noise[0])
    else:
        action[2] = abs(noise[0])
    if noise[1] < 0:
        action[3] = abs(noise[1])
    else:
        action[4] = abs(noise[1])

    action = np.clip(action, 0, 1.0)

    return action

env = mpe.env(render_mode="human")

env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(observation, reward, termination, truncation, info)

    if termination or truncation:
        action = None
    else:
        # action = env.action_space(agent).sample() # this is where you would insert your policy
        action = random_act()
        # action = np.array([0.001, 0.2, 0., 0., 0.], dtype=np.float32)
        # action = np.clip(action, 0, 1.0)

    env.step(action)

    time.sleep(0.1)


env.close()

# import math
#
# a = math.log2(0.05)
#
# l2n = a / 5000
#
# n = math.pow(2, l2n)
#
# print(n)
