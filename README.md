# Deep Reinforcement Learning Models for a Drone Hovering Task
Welcome to the repository of my Bachelor's thesis, which was conducted at the Department of Electrical and Computer Engineering at [TUM](https://www.tum.de/). The topic of this thesis is "Sample Complexity Analysis of Transfer Learning for Deep Reinforcement Learning Models".

In context of this Bachelor's thesis, I implement different deep Reinforcement Learning models on a drone to perform a hovering task in a Transfer Learning setting; first I pre-train the Reinforcement Learning models on the differential equations-based simulation environment of the drone where an amount of knowledge is acquired, then I transfer this knowledge by post-training the models on a [PyBullet](https://github.com/bulletphysics/bullet3)-based 3D simulation environment. By means of commonly used metrics, I evaluate the benefits of Transfer Learning to the learning process of the desired task compared to the case where no knowledge is transferred. Furthermore and most importantly, I analyze the sample complexity of the post-training of the implemented deep Reinforcement Learning algorithms. This allows me to draw conclusions about which models deliver the overall best performance and are most appropriate for the combination of deep Reinforcement Learning with Transfer Learning in this specific use case.

For a more detailed discussion please refer to [my thesis](https://github.com/Malekba98/Deep_Reinforcement_Learning_Bachelor_thesis/blob/main/report_and_presentation/report.pdf) and [final presentation](https://github.com/Malekba98/Deep_Reinforcement_Learning_Bachelor_thesis/blob/main/report_and_presentation/presentation.pdf).

This repository contains implementations of the differential equations-based and the [PyBullet](https://github.com/bulletphysics/bullet3)-based simulation environments used in this thesis, which were forked from a previous version of this [repository](https://github.com/SvenGronauer/phoenix-drone-simulation).

Hovering Task 
--- 
In the considered task, the drone is controlled to maintain its position at (0,0,1).

![Hover](./docs_readme/hover.png)

## Overview of Simulation Environments

|                                       | Task           | Physics            | Observation Frequency | Domain Randomization |  *Aerodynamic effects*  |
|-------------------------------------: | :----------:   | :----------------: | :-------------------: | :------------------: | :-------------------------: |
| `DroneHoverPIDSystemEqEnv-v0`         | Hover          | Differential Equations |  100 Hz |  10%        |   None |                 
| `DroneHoverPIDBulletEnv-v0`           | Hover          | PyBullet     |  100 Hz |        10%     |             Drag |                 




# Installation and Requirements
To clone and install the repository, run the following three lines:
```
$ git clone https://github.com/Malekba98/Deep_Reinforcement_Learning_Bachelor_thesis.git
$ cd Deep_Reinforcement_Learning_Bachelor_thesis/
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
