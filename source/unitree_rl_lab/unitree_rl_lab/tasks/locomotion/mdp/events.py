from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _initialize_platform_motion_state(
    env: ManagerBasedRLEnv,
    asset: RigidObject,
) -> None:
    """Lazily initialize cached platform base state and motion randomization buffers."""
    if not hasattr(env, "_platform_base_root_state_w"):
        default_root = asset.data.default_root_state.clone()
        default_root[:, :3] += env.scene.env_origins
        env._platform_base_root_state_w = default_root
    if not hasattr(env, "_platform_motion_phase"):
        env._platform_motion_phase = torch.zeros((env.num_envs, 6), device=env.device)
    if not hasattr(env, "_platform_motion_phase_target"):
        env._platform_motion_phase_target = 2.0 * math.pi * torch.rand((env.num_envs, 6), device=env.device)
    if not hasattr(env, "_platform_motion_frequency_hz"):
        env._platform_motion_frequency_hz = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_platform_motion_linear_acc"):
        env._platform_motion_linear_acc = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_platform_motion_angular_acc"):
        env._platform_motion_angular_acc = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_platform_motion_ramp_duration_s"):
        env._platform_motion_ramp_duration_s = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_platform_motion_phase_blend_duration_s"):
        env._platform_motion_phase_blend_duration_s = torch.zeros(env.num_envs, device=env.device)


def _sample_uniform(
    count: int,
    device: torch.device,
    low: float,
    high: float,
) -> torch.Tensor:
    """Sample ``count`` values uniformly in ``[low, high]``."""
    low = float(low)
    high = float(high)
    if high < low:
        low, high = high, low
    if abs(high - low) < 1.0e-8:
        return torch.full((count,), low, device=device)
    return low + (high - low) * torch.rand(count, device=device)


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    """Smoothly ramp a scalar from 0 to 1."""
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smoothstep01_derivative(x: torch.Tensor) -> torch.Tensor:
    """Derivative of smoothstep on the unit interval."""
    x_clamped = torch.clamp(x, 0.0, 1.0)
    derivative = 6.0 * x_clamped * (1.0 - x_clamped)
    return derivative * ((x >= 0.0) & (x <= 1.0)).float()


def _resample_platform_motion_episode_state(
    env: ManagerBasedRLEnv,
    env_ids_t: torch.Tensor,
    *,
    lin_frequency_hz: float,
    max_linear_acc: float,
    max_angular_acc: float,
    sample_frequency: bool,
    sample_linear_acc: bool,
    sample_angular_acc: bool,
    min_frequency_hz: float,
    min_linear_acc: float,
    min_angular_acc: float,
    startup_ramp_time_range_s: tuple[float, float],
    phase_blend_time_range_s: tuple[float, float],
) -> None:
    """Sample per-episode motion parameters for the specified environments."""
    count = len(env_ids_t)
    if count == 0:
        return

    if sample_frequency:
        env._platform_motion_frequency_hz[env_ids_t] = _sample_uniform(
            count, env.device, min_frequency_hz, lin_frequency_hz
        )
    else:
        env._platform_motion_frequency_hz[env_ids_t] = float(lin_frequency_hz)

    if sample_linear_acc:
        env._platform_motion_linear_acc[env_ids_t] = _sample_uniform(
            count, env.device, min_linear_acc, max_linear_acc
        )
    else:
        env._platform_motion_linear_acc[env_ids_t] = float(max_linear_acc)

    if sample_angular_acc:
        env._platform_motion_angular_acc[env_ids_t] = _sample_uniform(
            count, env.device, min_angular_acc, max_angular_acc
        )
    else:
        env._platform_motion_angular_acc[env_ids_t] = float(max_angular_acc)

    env._platform_motion_phase[env_ids_t] = 0.0
    env._platform_motion_phase_target[env_ids_t] = 2.0 * math.pi * torch.rand((count, 6), device=env.device)
    env._platform_motion_ramp_duration_s[env_ids_t] = _sample_uniform(
        count, env.device, startup_ramp_time_range_s[0], startup_ramp_time_range_s[1]
    )
    env._platform_motion_phase_blend_duration_s[env_ids_t] = _sample_uniform(
        count, env.device, phase_blend_time_range_s[0], phase_blend_time_range_s[1]
    )


def move_platform_sine(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    lin_frequency_hz: float = 0.2,
    max_linear_acc: float = 0.5,
    max_angular_acc: float = 0.125,
    sample_frequency: bool = False,
    sample_linear_acc: bool = False,
    sample_angular_acc: bool = False,
    min_frequency_hz: float = 0.0,
    min_linear_acc: float = 0.0,
    min_angular_acc: float = 0.0,
    startup_ramp_time_range_s: tuple[float, float] = (0.3, 0.5),
    phase_blend_time_range_s: tuple[float, float] = (0.3, 0.5),
):
    """Move platform with sinusoidal motion and curriculum-controlled DoF/amplitude.

    Notes:
        - The configured ``lin_frequency_hz`` is treated as the reference frequency for converting
          acceleration bounds to displacement amplitudes. This keeps low sampled frequencies from
          creating unbounded displacements.
        - Sampled per-episode frequency controls temporal speed, while sampled acceleration controls
          the displacement envelope.
        - DoF and amplitude scale are read from environment attributes set by curriculum.
        - ``env.platform_motion_mode`` selects one of four training modes.
        - ``env.platform_motion_level`` progressively enables DoFs within that mode.
    """
    if len(env_ids) == 0:
        return

    asset: RigidObject = env.scene[asset_cfg.name]
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    # Lazy initialization of cached state and per-env phase.
    _initialize_platform_motion_state(env, asset)
    missing_state_env_ids = env_ids_t[env._platform_motion_ramp_duration_s[env_ids_t] <= 0.0]
    if len(missing_state_env_ids) > 0:
        _resample_platform_motion_episode_state(
            env,
            missing_state_env_ids,
            lin_frequency_hz=lin_frequency_hz,
            max_linear_acc=max_linear_acc,
            max_angular_acc=max_angular_acc,
            sample_frequency=sample_frequency,
            sample_linear_acc=sample_linear_acc,
            sample_angular_acc=sample_angular_acc,
            min_frequency_hz=min_frequency_hz,
            min_linear_acc=min_linear_acc,
            min_angular_acc=min_angular_acc,
            startup_ramp_time_range_s=startup_ramp_time_range_s,
            phase_blend_time_range_s=phase_blend_time_range_s,
        )

    level = int(getattr(env, "platform_motion_level", 1))
    amp_scale = float(getattr(env, "platform_motion_amp_scale", 0.1))
    amp_scale = max(0.0, min(1.0, amp_scale))

    motion_mode = str(getattr(env, "platform_motion_mode", "rpy"))
    dof_mask = _platform_motion_dof_mask(motion_mode, level, env.device)

    t = env.episode_length_buf[env_ids_t].float() * env.step_dt
    sampled_frequency_hz = env._platform_motion_frequency_hz[env_ids_t]
    sampled_linear_acc = env._platform_motion_linear_acc[env_ids_t]
    sampled_angular_acc = env._platform_motion_angular_acc[env_ids_t]

    reference_omega = 2.0 * math.pi * max(float(lin_frequency_hz), 1.0e-4)
    reference_omega_sq = reference_omega * reference_omega
    omega = 2.0 * math.pi * sampled_frequency_hz

    # Convert sampled acceleration bounds to amplitudes using the configured max frequency as reference.
    max_lin_amp = sampled_linear_acc / reference_omega_sq
    max_ang_amp = sampled_angular_acc / reference_omega_sq

    ramp_duration = torch.clamp(env._platform_motion_ramp_duration_s[env_ids_t], min=1.0e-6)
    phase_blend_duration = torch.clamp(env._platform_motion_phase_blend_duration_s[env_ids_t], min=1.0e-6)
    ramp_progress = t / ramp_duration
    ramp_gain = _smoothstep01(ramp_progress) * amp_scale
    ramp_gain_dot = _smoothstep01_derivative(ramp_progress) * (amp_scale / ramp_duration)

    phase_progress = t / phase_blend_duration
    phase_alpha = _smoothstep01(phase_progress).unsqueeze(-1)
    phase_alpha_dot = (_smoothstep01_derivative(phase_progress) / phase_blend_duration).unsqueeze(-1)
    target_phase = env._platform_motion_phase_target[env_ids_t]
    phase = phase_alpha * target_phase
    env._platform_motion_phase[env_ids_t] = phase

    motion_enabled = (sampled_frequency_hz > 1.0e-4).float()
    lin_amp = max_lin_amp * ramp_gain * motion_enabled
    ang_amp = max_ang_amp * ramp_gain * motion_enabled
    lin_amp_dot = max_lin_amp * ramp_gain_dot * motion_enabled
    ang_amp_dot = max_ang_amp * ramp_gain_dot * motion_enabled

    amp_vec = torch.stack(
        [
            lin_amp,
            0.8 * lin_amp,
            0.5 * lin_amp,
            ang_amp,
            0.8 * ang_amp,
            0.6 * ang_amp,
        ],
        dim=-1,
    )
    amp_vec_dot = torch.stack(
        [
            lin_amp_dot,
            0.8 * lin_amp_dot,
            0.5 * lin_amp_dot,
            ang_amp_dot,
            0.8 * ang_amp_dot,
            0.6 * ang_amp_dot,
        ],
        dim=-1,
    )

    arg = omega.unsqueeze(-1) * t.unsqueeze(-1) + phase

    sin_part = torch.sin(arg)
    cos_part = torch.cos(arg)
    offsets = sin_part * amp_vec * dof_mask.unsqueeze(0)

    base_root = env._platform_base_root_state_w[env_ids_t]
    pos_w = base_root[:, :3] + offsets[:, :3]

    quat_delta = quat_from_euler_xyz(offsets[:, 3], offsets[:, 4], offsets[:, 5])
    quat_w = quat_mul(base_root[:, 3:7], quat_delta)

    asset.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=env_ids_t)

    vel = torch.zeros((len(env_ids_t), 6), device=env.device)
    arg_dot = omega.unsqueeze(-1) + phase_alpha_dot * target_phase
    vel[:, :3] = (
        amp_vec_dot[:, :3] * sin_part[:, :3] + amp_vec[:, :3] * cos_part[:, :3] * arg_dot[:, :3]
    ) * dof_mask[:3].unsqueeze(0)
    vel[:, 3:] = (
        amp_vec_dot[:, 3:] * sin_part[:, 3:] + amp_vec[:, 3:] * cos_part[:, 3:] * arg_dot[:, 3:]
    ) * dof_mask[3:].unsqueeze(0)
    asset.write_root_velocity_to_sim(vel, env_ids=env_ids_t)


def reset_platform_state(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    lin_frequency_hz: float = 0.2,
    max_linear_acc: float = 0.5,
    max_angular_acc: float = 0.125,
    sample_frequency: bool = False,
    sample_linear_acc: bool = False,
    sample_angular_acc: bool = False,
    min_frequency_hz: float = 0.0,
    min_linear_acc: float = 0.0,
    min_angular_acc: float = 0.0,
    startup_ramp_time_range_s: tuple[float, float] = (0.3, 0.5),
    phase_blend_time_range_s: tuple[float, float] = (0.3, 0.5),
):
    """Reset platform pose/velocity to default state and resample its phase.

    这样可以保证每个 episode 的起点都是“平台先回到中性位姿，再生成机器人”，
    避免当平台启用了 z/roll/pitch 自由度时，机器人按旧高度生成到平台内部。
    """
    if len(env_ids) == 0:
        return

    asset: RigidObject = env.scene[asset_cfg.name]
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    _initialize_platform_motion_state(env, asset)

    # 将平台直接恢复到默认根状态对应的世界系位姿。
    base_root = env._platform_base_root_state_w[env_ids_t]
    asset.write_root_pose_to_sim(base_root[:, :7], env_ids=env_ids_t)
    asset.write_root_velocity_to_sim(torch.zeros((len(env_ids_t), 6), device=env.device), env_ids=env_ids_t)

    # Reset 时平台从中性位姿和零相位起步，同时为后续平滑过渡采样目标随机相位与运动参数。
    _resample_platform_motion_episode_state(
        env,
        env_ids_t,
        lin_frequency_hz=lin_frequency_hz,
        max_linear_acc=max_linear_acc,
        max_angular_acc=max_angular_acc,
        sample_frequency=sample_frequency,
        sample_linear_acc=sample_linear_acc,
        sample_angular_acc=sample_angular_acc,
        min_frequency_hz=min_frequency_hz,
        min_linear_acc=min_linear_acc,
        min_angular_acc=min_angular_acc,
        startup_ramp_time_range_s=startup_ramp_time_range_s,
        phase_blend_time_range_s=phase_blend_time_range_s,
    )


def _platform_motion_dof_mask(mode: str, level: int, device: torch.device) -> torch.Tensor:
    """Return DoF mask for a platform training mode and curriculum level.

    DoF order is ``[x, y, z, roll, pitch, yaw]``.
    """
    dof_mask = torch.zeros(6, device=device)
    level = max(0, level)

    if level <= 0:
        return dof_mask

    if mode == "rpy":
        # Rotation-only mode: progressively enable roll, pitch, yaw.
        dof_mask[3 : 3 + min(level, 3)] = 1.0
        return dof_mask
    if mode == "xyz":
        # Translation-only mode: progressively enable x, y, z.
        dof_mask[: min(level, 3)] = 1.0
        return dof_mask
    if mode == "z_rp":
        # Three-DoF mode: progressively enable z, roll, pitch.
        active_ids = [2, 3, 4]
        dof_mask[active_ids[: min(level, 3)]] = 1.0
        return dof_mask
    if mode == "full":
        # Six-DoF mode: progressively enable x, y, z, roll, pitch, yaw.
        dof_mask[: min(level, 6)] = 1.0
        return dof_mask

    raise ValueError(f"Unsupported platform motion mode: {mode}")


def get_platform_motion_mode_max_level(mode: str) -> int:
    """Return the maximum curriculum level for a platform motion mode."""
    if mode in ("rpy", "xyz", "z_rp"):
        return 3
    if mode == "full":
        return 6
    raise ValueError(f"Unsupported platform motion mode: {mode}")


def get_platform_motion_mode_description(mode: str) -> str:
    """Return a short human-readable description for a platform motion mode."""
    if mode == "rpy":
        return "roll/pitch/yaw rotation only"
    if mode == "xyz":
        return "x/y/z translation only"
    if mode == "z_rp":
        return "z translation with roll/pitch rotation"
    if mode == "full":
        return "full 6-DoF translation and rotation"
    raise ValueError(f"Unsupported platform motion mode: {mode}")
