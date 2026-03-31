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
    dof_upgrade_every_episodes: int = 140, # 平板运动难度升级的频率（以训练的episode数量为单位）
    amp_ramp_episodes: int = 800, # 平板运动幅度提升的时间长度（以训练的episode数量为单位）
    min_amp_scale: float = 0.1, # 平板运动幅度的最小缩放比例，确保即使在训练初期也有一定的运动挑战
    stationary_episodes: int = 200, # 热身期：前N个episode平台保持完全静止
    min_episodes_per_level: int = 80,
    base_quality_threshold: float = 0.72,
    per_level_quality_increment: float = 0.02,
    quality_ema_alpha: float = 0.90,
) -> torch.Tensor:
    """Curriculum for platform motion complexity.

    - Warm-up: keep platform stationary for ``stationary_episodes``.
    - DoF level: starts at 1 and increases to 6 after warm-up.
    - Amplitude scale: starts small and ramps to 1.0 after warm-up.
    """
    if not hasattr(env, "platform_motion_level"):
        env.platform_motion_level = 0
    if not hasattr(env, "platform_motion_amp_scale"):
        env.platform_motion_amp_scale = min_amp_scale
    if not hasattr(env, "platform_level_start_episode"):
        env.platform_level_start_episode = 0
    if not hasattr(env, "platform_curriculum_score"):
        env.platform_curriculum_score = 0.0
    if not hasattr(env, "platform_curriculum_score_ema"):
        env.platform_curriculum_score_ema = 0.0

    if env.common_step_counter % env.max_episode_length == 0:
        episode_count = int(env.common_step_counter // env.max_episode_length)
        if episode_count < stationary_episodes:
            # During warm-up, force platform to remain static.
            env.platform_motion_level = 0
            env.platform_motion_amp_scale = 0.0
            env.platform_level_start_episode = episode_count
        else:
            active_episode_count = episode_count - stationary_episodes
            progress = min(1.0, active_episode_count / max(1, amp_ramp_episodes))
            env.platform_motion_amp_scale = min_amp_scale + (1.0 - min_amp_scale) * progress

            target_level_from_count = max(1, min(6, 1 + active_episode_count // dof_upgrade_every_episodes))
            if env.platform_motion_level <= 0:
                env.platform_motion_level = 1
                env.platform_level_start_episode = episode_count

            score = _platform_curriculum_quality_score(env, env_ids)
            env.platform_curriculum_score = score
            env.platform_curriculum_score_ema = (
                quality_ema_alpha * float(env.platform_curriculum_score_ema) + (1.0 - quality_ema_alpha) * score
            )

            level_dwell_episodes = episode_count - int(env.platform_level_start_episode)
            level_based_threshold = min(
                0.90,
                base_quality_threshold + per_level_quality_increment * max(0, int(env.platform_motion_level) - 1),
            )
            count_gate = int(env.platform_motion_level) < int(target_level_from_count)
            dwell_gate = level_dwell_episodes >= min_episodes_per_level
            quality_gate = float(env.platform_curriculum_score_ema) >= level_based_threshold

            if count_gate and dwell_gate and quality_gate:
                env.platform_motion_level = min(6, int(env.platform_motion_level) + 1)
                env.platform_level_start_episode = episode_count

    return torch.tensor(float(env.platform_motion_level), device=env.device)


def _platform_curriculum_quality_score(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> float:
    """Compute stage progression quality in [0, 1] from normalized reward-term performance."""
    if len(env_ids) == 0:
        ids = slice(None)
    else:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    def _term_score(term_name: str) -> tuple[bool, float]:
        try:
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            term_weight = float(term_cfg.weight)
            term_rate = (
                torch.mean(env.reward_manager._episode_sums[term_name][ids]) / max(1e-6, float(env.max_episode_length_s))
            )
            rate_val = float(term_rate)
        except Exception:
            return False, 0.0

        if term_weight > 0.0:
            return True, float(max(0.0, min(1.0, rate_val / max(1e-6, term_weight))))
        if term_weight < 0.0:
            # For penalties, closer to 0 is better.
            return True, float(max(0.0, min(1.0, 1.0 + rate_val / max(1e-6, abs(term_weight)))))
        return True, 0.0

    metrics = [
        ("track_lin_vel_xy", 0.40),
        ("track_ang_vel_z", 0.25),
        ("alive", 0.20),
        ("relative_platform_velocity", 0.15),
    ]
    weighted_sum = 0.0
    total_weight = 0.0
    for term_name, metric_weight in metrics:
        has_term, score = _term_score(term_name)
        if has_term:
            weighted_sum += metric_weight * score
            total_weight += metric_weight
    if total_weight <= 1e-6:
        return 0.0
    return weighted_sum / total_weight


def _set_reward_term_weight(env: ManagerBasedRLEnv, term_name: str, weight: float) -> None:
    """Update reward term weight in-place."""
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = float(weight)


def platform_reward_weight_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> torch.Tensor:
    """Adjust reward weights according to platform motion curriculum level.

    Design:
    - level 0 (platform stationary): use stronger, flat-ground-like posture penalties.
    - level 1-2: mildly relax posture penalties.
    - level 3-4: further relax posture penalties.
    - level 5-6: keep the most relaxed posture penalties to avoid over-penalizing platform-induced motion.
    """
    # Read current platform motion level set by platform_motion_levels().
    level = int(getattr(env, "platform_motion_level", 0))
    level = max(0, min(6, level))

    # Update only once per episode boundary to avoid per-step churn.
    if env.common_step_counter % env.max_episode_length == 0:
        if level <= 0:
            # Flat-ground-like stage: encourage learning to walk first.
            weights = {
                "track_lin_vel_xy": 2.0,
                "track_ang_vel_z": 1.0,
                "base_linear_velocity": -1.0,
                "base_angular_velocity": -0.05,
                "flat_orientation_l2": -2.0,
                "base_height": -5.0,
            }
            profile_id = 0.0
        elif level <= 2:
            weights = {
                "track_lin_vel_xy": 2.0,
                "track_ang_vel_z": 1.0,
                "base_linear_velocity": -0.5,
                "base_angular_velocity": -0.03,
                "flat_orientation_l2": -1.5,
                "base_height": -3.0,
            }
            profile_id = 1.0
        elif level <= 4:
            weights = {
                "track_lin_vel_xy": 2.0,
                "track_ang_vel_z": 1.0,
                "base_linear_velocity": -0.3,
                "base_angular_velocity": -0.025,
                "flat_orientation_l2": -1.2,
                "base_height": -2.5,
            }
            profile_id = 2.0
        else:
            weights = {
                "track_lin_vel_xy": 2.0,
                "track_ang_vel_z": 1.0,
                "base_linear_velocity": -0.2,
                "base_angular_velocity": -0.02,
                "flat_orientation_l2": -1.0,
                "base_height": -2.0,
            }
            profile_id = 3.0

        for term_name, weight in weights.items():
            _set_reward_term_weight(env, term_name, weight)

        env.platform_reward_profile_id = profile_id

    return torch.tensor(float(getattr(env, "platform_reward_profile_id", 0.0)), device=env.device)


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


def platform_curriculum_score(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Log instant curriculum quality score in [0, 1]."""
    score = float(getattr(env, "platform_curriculum_score", 0.0))
    return torch.tensor(score, device=env.device)


def platform_curriculum_score_ema(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Log EMA-smoothed curriculum quality score in [0, 1]."""
    score = float(getattr(env, "platform_curriculum_score_ema", 0.0))
    return torch.tensor(score, device=env.device)
