from __future__ import annotations

import torch
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .events import get_platform_motion_mode_max_level

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
    motion_mode: str = "xyz", # 平台运动模式，支持 "rpy"（逐步增加滚转、俯仰、偏航）、"xyz"（逐步增加x、y、z平移）、"z_rp"（逐步增加z平移和滚转、俯仰）和 "full"（逐步增加全部6个DoF）
    dof_upgrade_every_episodes: int = 140, # 平板运动难度升级的频率（以训练的episode数量为单位）
    amp_ramp_episodes: int = 800, # 平板运动幅度提升的时间长度（以训练的episode数量为单位）
    min_amp_scale: float = 0.1, # 平板运动幅度的最小缩放比例，确保即使在训练初期也有一定的运动挑战
    stationary_episodes: int = 200, # 热身期：前N个episode平台保持完全静止
    min_episodes_per_level: int = 80,
    base_quality_threshold: float = 0.72,
    per_level_quality_increment: float = 0.02,
    quality_ema_alpha: float = 0.90,
    max_motion_level: int | None = None,
) -> torch.Tensor:
    """Curriculum for platform motion complexity.

    - Warm-up: keep platform stationary for ``stationary_episodes``.
    - Training mode is selected by ``motion_mode``.
    - DoF level starts at 1 and increases to the mode-specific maximum after warm-up.
    - Amplitude scale: starts small and ramps to 1.0 after warm-up.
    """
    mode_max_level = get_platform_motion_mode_max_level(motion_mode)
    if max_motion_level is None:
        effective_max_level = mode_max_level
    else:
        effective_max_level = max(0, min(int(max_motion_level), mode_max_level))

    if not hasattr(env, "platform_motion_level"):
        env.platform_motion_level = 0
    if not hasattr(env, "platform_motion_amp_scale"):
        env.platform_motion_amp_scale = min_amp_scale
    if not hasattr(env, "platform_motion_mode"):
        env.platform_motion_mode = motion_mode
    if not hasattr(env, "platform_level_start_episode"):
        env.platform_level_start_episode = 0
    if not hasattr(env, "platform_curriculum_score"):
        env.platform_curriculum_score = 0.0
    if not hasattr(env, "platform_curriculum_score_ema"):
        env.platform_curriculum_score_ema = 0.0

    env.platform_motion_mode = motion_mode

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

            target_level_from_count = max(1, min(effective_max_level, 1 + active_episode_count // dof_upgrade_every_episodes))
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
                env.platform_motion_level = min(effective_max_level, int(env.platform_motion_level) + 1)
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


def _lerp(start: float, end: float, alpha: float) -> float:
    """线性插值辅助函数。"""
    alpha = max(0.0, min(1.0, float(alpha)))
    return (1.0 - alpha) * float(start) + alpha * float(end)


def _get_enabled_platform_dofs(mode: str, level: int) -> dict[str, bool]:
    """根据当前训练模式和课程级别，返回实际已启用的平台 DoF。

    返回字典的键固定为 ``x/y/z/roll/pitch/yaw``，值为布尔量。

    这里故意不直接按“模式名”给奖励偏置，而是先展开成具体 DoF：
    - 后续维护时，只需要关注“某个自由度会影响哪些奖励”；
    - 即使以后新增模式，只要最终能映射到这些 DoF，奖励调度逻辑就不需要重写。
    """
    level = max(0, int(level))
    enabled = {
        "x": False,
        "y": False,
        "z": False,
        "roll": False,
        "pitch": False,
        "yaw": False,
    }

    if level <= 0:
        return enabled

    if mode == "rpy":
        order = ["roll", "pitch", "yaw"]
    elif mode == "xyz":
        order = ["x", "y", "z"]
    elif mode == "z_rp":
        order = ["z", "roll", "pitch"]
    elif mode == "full":
        order = ["x", "y", "z", "roll", "pitch", "yaw"]
    else:
        raise ValueError(f"Unsupported platform motion mode: {mode}")

    for dof_name in order[:level]:
        enabled[dof_name] = True
    return enabled


def platform_reward_weight_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> torch.Tensor:
    """根据平台运动幅度与已启用的具体 DoF，连续调整奖励权重。

    设计目标：
    1. 平台运动幅度越大，越容易把“由平台引起的被动运动”误判为机器人自身的不稳定，
       因此需要连续放松部分惩罚项，而不是只按 level 做硬分段。
    2. 不同 DoF 对奖励的影响方向不同，因此偏置应当按实际启用的 xyzrpy 来决定，
       而不是简单按某个模式名一刀切。
    3. 速度跟踪类奖励（``track_lin_vel_xy`` / ``track_ang_vel_z``）保留稳定权重，
       主要调整那些最容易被平台运动“误伤”的姿态、速度和高度惩罚项。
    """
    level = int(getattr(env, "platform_motion_level", 0))
    amp_scale = float(getattr(env, "platform_motion_amp_scale", 0.0))
    motion_mode = str(getattr(env, "platform_motion_mode", "xyz"))

    # 平台运动幅度缩放系数来自 curriculum：
    # 0.0 表示平台静止或几乎静止；
    # 1.0 表示达到当前训练计划允许的最大振幅。
    # 这里将其直接作为“奖励放松强度”的主控制量，做连续插值。
    relax = max(0.0, min(1.0, amp_scale))

    # 根据当前 mode + level 推导出“此刻到底启用了哪些自由度”。
    # 后面的偏置全部建立在这个字典上，这样逻辑更贴近物理含义。
    enabled_dofs = _get_enabled_platform_dofs(motion_mode, level)

    # Update only once per episode boundary to avoid per-step churn.
    if env.common_step_counter % env.max_episode_length == 0:
        # 第一层：只根据平台幅度做连续插值。
        #
        # 这些起止值延续了你之前的设计意图：
        # - amp_scale 小时，更接近平地训练，惩罚更强；
        # - amp_scale 大时，更接近强扰动平台，惩罚更宽松。
        #
        # 注意：这里仍保留 track 奖励为常数，是为了避免平台扰动一变大，
        # 机器人就失去对速度跟踪任务本身的学习驱动力。
        weights = {
            "track_lin_vel_xy": 2.0,
            "track_ang_vel_z": 1.0,
            "base_linear_velocity": _lerp(-1.0, -0.2, relax),
            "base_angular_velocity": _lerp(-0.05, -0.02, relax),
            "flat_orientation_l2": _lerp(-2.0, -1.0, relax),
            "base_height": _lerp(-5.0, -0.2, relax),
        }

        # 第二层：根据“实际启用的自由度”做偏置修正。
        #
        # 这里的原则是：
        # - 哪个自由度最可能直接扰动某个状态量，就优先放松该状态对应的惩罚；
        # - 偏置是在连续插值结果上再乘一个系数，而不是重新指定一套离散权重，
        #   这样幅度和 DoF 的影响是可组合的。
        #
        # 系数小于 1 表示“进一步放松惩罚”（负权重绝对值变小，更接近 0）。
        # 之所以不把系数做得过小，是为了避免奖励完全失去约束作用。

        # z 自由度会直接改变平台法向高度，因此最容易误伤 base_height。
        # 当启用 z 时，对高度惩罚做大幅放松；同时稍微放松线速度惩罚，
        # 因为上下运动会间接放大机身线速度波动。
        if enabled_dofs["z"]:
            weights["base_height"] *= 0.35
            weights["base_linear_velocity"] *= 0.85

        # x/y 平移会让机器人在世界系中出现被动平移速度，因此主要放松线速度惩罚。
        # y 向和 x 向分开判断，是为了保留以后做各向异性调参的空间；
        # 当前先给相同幅度的偏置，便于理解。
        if enabled_dofs["x"]:
            weights["base_linear_velocity"] *= 0.70
        if enabled_dofs["y"]:
            weights["base_linear_velocity"] *= 0.70

        # roll/pitch 会直接改变平台法向和机体姿态，是最容易误伤
        # flat_orientation_l2 和 base_angular_velocity 的自由度。
        # 同时，平台倾斜后“世界系高度”本身也更不稳定，所以对 base_height
        # 也做中等放松，避免机器人因为跟着平台倾斜而被过罚。
        if enabled_dofs["roll"]:
            weights["flat_orientation_l2"] *= 0.60
            weights["base_angular_velocity"] *= 0.75
            weights["base_height"] *= 0.85
        if enabled_dofs["pitch"]:
            weights["flat_orientation_l2"] *= 0.60
            weights["base_angular_velocity"] *= 0.75
            weights["base_height"] *= 0.85

        # yaw 主要带来平面内朝向变化，对“平地姿态”惩罚影响不如 roll/pitch 明显，
        # 但会显著增加机体角速度响应，因此主要放松角速度惩罚，少量放松线速度惩罚。
        if enabled_dofs["yaw"]:
            weights["base_angular_velocity"] *= 0.75
            weights["base_linear_velocity"] *= 0.90

        # 最后给几个下限，防止多种 DoF 同时启用时惩罚项被乘得过小，
        # 造成奖励失去约束、训练发散或策略钻空子。
        weights["base_linear_velocity"] = min(weights["base_linear_velocity"], -0.08)
        weights["base_angular_velocity"] = min(weights["base_angular_velocity"], -0.01)
        weights["flat_orientation_l2"] = min(weights["flat_orientation_l2"], -0.30)
        weights["base_height"] = min(weights["base_height"], -0.1)

        for term_name, weight in weights.items():
            _set_reward_term_weight(env, term_name, weight)

        # 这里不再使用离散 profile id，而是记录一个连续值用于日志：
        # 整数部分编码当前已启用 DoF 数量，小数部分编码 amp_scale。
        # 这样在 TensorBoard 或调试输出中，一眼就能看出“当前放松到了什么程度”。
        enabled_count = sum(int(flag) for flag in enabled_dofs.values())
        env.platform_reward_profile_id = float(enabled_count) + 0.01 * round(100.0 * relax)

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
