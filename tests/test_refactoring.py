"""
Copyright (c) 2022 Sven Gronauer (Technical University of Munich)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from phoenix_simulation.envs.DroneHover import DroneHoverPWMBulletEnv
import time
import numpy as np


if __name__ == '__main__':
    env = DroneHoverPWMBulletEnv()
    env.render()
    env.reset()
    done = False
    while not done:
        for i in range(100):
            action = np.ones_like(env.action_space.sample())
            x, r, done, info = env.step(action)
            time.sleep(0.01)

