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

