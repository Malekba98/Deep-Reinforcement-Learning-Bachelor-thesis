# Deep Reinforcement Learning Models for a Hovering Task
Welcome to the repository of my Bachelor's thesis at the Department of Electrical and Computer Engineering at TUM. The topic of this thesis is "Sample Complexity Analysis of Transfer Learning for Deep Reinforcement Learning Models".

In context of this Bachelor's thesis, I implement different deep Reinforcement Learning models on a drone to perform a hovering task in a Transfer Learning setting; first I pre-train the Reinforcement Learning models on the differential equations of the drone where an amount of knowledge is acquired, then I transfer this knowledge by post-training the models on a 3D simulation environment. By means of commonly used metrics, I evaluate the benefits of Transfer Learning to the learning process of the desired task compared to the case where no knowledge is transferred. Furthermore and most importantly, I analyze the sample complexity of the post-training of the implemented deep Reinforcement Learning algorithms. This allows me to draw conclusions about which models deliver the overall best performance and are most appropriate for the combination of deep Reinforcement Learning with Transfer Learning in this specific use case.

For a more detailed discussion please refer to my thesis and final presentation.

An OpenAI [Gym environment](https://gym.openai.com/envs/#classic_control) based on [PyBullet](https://github.com/bulletphysics/bullet3) for reinforcement learning with quadcopters. 

- The default dynamics are based on [Bitcraze's Crazyflie 2.x nano-quadrotor](https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf)

- Everything after a `$` is entered on a terminal, everything after `>>>` is passed to a Python interpreter



At the moment, there is one task available to fly the drone:

- Hover



## Overview of Environments

|                                       | Task         | Controller    | Physics            | Observation Frequency | Domain Randomization |  *Aerodynamic effects*  |
|-------------------------------------: | :----------: | :-----------: | :----------------: | :-------------------: | :------------------: | :-------------------------: |
| `DroneHoverPWMSystemEqEnv-v0`         | Hover        | PWM (100Hz)   | System Equations   | 100 Hz                | 10%                  | None |
| `DroneHoverPWMBulletEnv-v0`           | Hover        | PWM (500Hz)   | PyBullet           | 100 Hz                | 10%                  | Drag |
| `DroneHoverPIDSystemEqEnv-v0`         | Hover        | Attitude Rate PID controller (500Hz)   | System Equations |  100 Hz |  10%        |   None |                 
| `DroneHoverPIDBulletEnv-v0`           | Hover        | Attitude Rate PID controller (500Hz)   | PyBullet     |  100 Hz |        10%     |             Drag |                 




# Installation and Requirements

Here are the (few) steps to follow to get our repository ready to run. Clone the
repository and install the phoenix-simulation package via pip. Use the following
three lines:

The repo is structured as a [Gym Environment](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
and can be installed with `pip install --editable`
```
$ git clone https://gitlab.lrz.de/projectphoenix/phoenix-simulation.git
$ cd phoenix-simulation/
$ pip install -e .
```

> Note: if your default `python` is 2.7, in the following, replace `pip` with `pip3` and `python` with `python3`


## Supported Systems

We tested the repository under *Ubuntu 20.04* and *Mac OS X 11.2* running Python 3.7
and 3.8. Other system might work as well but have not been tested yet.
Note that PyBullet supports Windows as platform only experimentally!. 

Note: This package has been tested on Mac OS 11 and Ubuntu (18.04 LTS, 
20.04 LTS), and is probably fine for most recent Mac and Linux operating 
systems. 


## Dependencies 

Bullet-Safety-Gym heavily depends on two packages:

+ [Gym](https://github.com/openai/gym)
+ [PyBullet](https://github.com/bulletphysics/bullet3)


## Getting Started


After the successful installation of the repository, the Bullet-Safety-Gym 
environments can be simply instantiated via `gym.make`. See: 

```
>>> import gym
>>> import phoenix_simulation
>>> env = gym.make('DroneHoverPWMBulletEnv-v0')
```

The functional interface follows the API of the OpenAI Gym (Brockman et al., 
2016) that consists of the three following important functions:

```
>>> observation = env.reset()
>>> random_action = env.action_space.sample()  # usually the action is determined by a policy
>>> next_observation, reward, done, info = env.step(random_action)
```

Besides the reward signal, our environments provide additional information 
that is contained in the `info` dictionary:
```
>>> info
{'cost': 1.0}
```

A minimal code for visualizing a uniformly random policy in a GUI, can be seen 
in:

```
import gym
import phoenix_simulation

>>> env = gym.make('DroneHoverPWMBulletEnv-v0')

while True:
    done = False
    env.render()  # make GUI of PyBullet appear
    x = env.reset()
    while not done:
        random_action = env.action_space.sample()
        x, reward, done, info = env.step(random_action)
```
Note that only calling the render function before the reset function triggers 
visuals.


# Examples

- `generate_trajectories.py` @ Matthias Kissel

See the `generate_trajectories.py` script which shows how to generates a data 
batch of size N. Use `generate_trajectories.py --play` to visualize the policy
in PyBullet simulator. 

# Tools

- `convert.py` @ Sven Gronauer

A function used by Sven to extract the policy networks from
his trained Actor Critic module and convert the model to a json file format.




# To-Dos

- Implementation of Take-Off, Reach and Circle tasks

# Version History and Changes


| Version | Changes | Date |
|-------: | :----------------: |  :----------------: |
| 1.1     | Re-factoring of repository  (only Hover task yet implemented)  | 18.05.2021 | 
| 1.0     | Fork from [Gym-PyBullet-Drones Repo](https://github.com/utiasDSL/gym-pybullet-drones)  | 01.12.2020 | 


-----

> Technical University of Munich [Chair of Data Processing](https://www.ei.tum.de/en/ldv/homepage/)

-----
A big thanks goes to:
- [Gym-PyBullet-Drones Repo](https://github.com/utiasDSL/gym-pybullet-drones) 
  which was the staring point for this repository.
  
- Jakob Foerster - Bachelor Thesis (for the parameter values)
