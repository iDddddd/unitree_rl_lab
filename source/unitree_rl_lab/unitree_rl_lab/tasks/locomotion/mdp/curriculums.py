from __future__ import annotations

import torch
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def platform_motion_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    dof_upgrade_every_episodes: int = 100, # 平板运动难度升级的频率（以训练的episode数量为单位）
    amp_ramp_episodes: int = 500, # 平板运动幅度提升的时间长度（以训练的episode数量为单位）
    min_amp_scale: float = 0.1, # 平板运动幅度的最小缩放比例，确保即使在训练初期也有一定的运动挑战
    stationary_episodes: int = 120, # 热身期：前N个episode平台保持完全静止
) -> torch.Tensor:
    """Curriculum for platform motion complexity.

    - Warm-up: keep platform stationary for ``stationary_episodes``.
    - DoF level: starts at 1 and increases to 6 after warm-up.
    - Amplitude scale: starts small and ramps to 1.0 after warm-up.
    """
    if not hasattr(env, "platform_motion_level"):
        env.platform_motion_level = 1
    if not hasattr(env, "platform_motion_amp_scale"):
        env.platform_motion_amp_scale = min_amp_scale

    if env.common_step_counter % env.max_episode_length == 0:
        episode_count = int(env.common_step_counter // env.max_episode_length)
        if episode_count < stationary_episodes:
            # During warm-up, force platform to remain static.
            env.platform_motion_level = 0
            env.platform_motion_amp_scale = 0.0
        else:
            active_episode_count = episode_count - stationary_episodes
            env.platform_motion_level = max(1, min(6, 1 + active_episode_count // dof_upgrade_every_episodes))

            progress = min(1.0, active_episode_count / max(1, amp_ramp_episodes))
            env.platform_motion_amp_scale = min_amp_scale + (1.0 - min_amp_scale) * progress

    return torch.tensor(float(env.platform_motion_level), device=env.device)


def platform_motion_amplitude(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_linear_acc: float = 0.5,
    lin_frequency_hz: float = 0.2,
) -> torch.Tensor:
    """Log current platform x-axis amplitude (meters) under acceleration bound."""
    amp_scale = float(getattr(env, "platform_motion_amp_scale", 0.1))
    omega = 2.0 * math.pi * lin_frequency_hz
    max_lin_amp = max_linear_acc / max(omega * omega, 1e-6)
    return torch.tensor(max_lin_amp * amp_scale, device=env.device)


def episode_count(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Log mean progress of current episode in [0, 1]."""
    # episode_length_buf 是每个 env 在当前 episode 已走过的步数。
    # 除以 max_episode_length 后得到当前 episode 进度，再取 env 维度均值用于日志展示。
    if len(env_ids) == 0:
        progress = env.episode_length_buf.float() / max(1, env.max_episode_length)
    else:
        env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        progress = env.episode_length_buf[env_ids_t].float() / max(1, env.max_episode_length)
    return torch.mean(progress)


def platform_motion_level(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Log current platform motion DoF level set by curriculum."""
    level = float(getattr(env, "platform_motion_level", 0))
    return torch.tensor(level, device=env.device)


def platform_amp_scale(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Log current platform motion amplitude scale set by curriculum."""
    amp_scale = float(getattr(env, "platform_motion_amp_scale", 0.0))
    return torch.tensor(amp_scale, device=env.device)
