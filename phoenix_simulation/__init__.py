from gym.envs.registration import register


# ==== PWM Control ===
register(
    id='DroneHoverPWMSysEqEnv-v0',
    entry_point='phoenix_simulation.envs.DroneHover:DroneHoverPWMSysEqEnv',
    max_episode_steps=500,
)

register(
    id='DroneHoverPWMBulletEnv-v0',
    entry_point='phoenix_simulation.envs.DroneHover:DroneHoverPWMBulletEnv',
    max_episode_steps=500,
)

# ==== PID Attitude Rate Control ===
register(
    id='DroneHoverPIDSysEqEnv-v0',
    entry_point='phoenix_simulation.envs.DroneHover:DroneHoverPIDSysEqEnv',
    max_episode_steps=500,
)

register(
    id='DroneHoverPIDBulletEnv-v0',
    entry_point='phoenix_simulation.envs.DroneHover:DroneHoverPIDBulletEnv',
    max_episode_steps=500,
)

