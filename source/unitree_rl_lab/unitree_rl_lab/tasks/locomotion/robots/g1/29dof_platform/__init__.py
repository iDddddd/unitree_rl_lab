import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-PlatformVelocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.platformvelocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.platformvelocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:RoaPPORunnerCfg",
    },
)
