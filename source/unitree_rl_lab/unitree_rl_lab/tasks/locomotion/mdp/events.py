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
    """Lazily initialize cached platform base state and sinusoid phase."""
    if not hasattr(env, "_platform_base_root_state_w"):
        default_root = asset.data.default_root_state.clone()
        default_root[:, :3] += env.scene.env_origins
        env._platform_base_root_state_w = default_root
    if not hasattr(env, "_platform_motion_phase"):
        env._platform_motion_phase = 2.0 * math.pi * torch.rand((env.num_envs, 6), device=env.device)


def move_platform_sine(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    lin_frequency_hz: float = 0.2,
    max_linear_acc: float = 0.5,
    max_angular_acc: float = 0.125,
):
    """Move platform with sinusoidal motion and curriculum-controlled DoF/amplitude.

    Notes:
        - Linear acceleration bound: ``a_max = A * (2*pi*f)^2 <= max_linear_acc``.
        - Angular acceleration bound uses the same sinusoidal form in radians.
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

    level = int(getattr(env, "platform_motion_level", 1))
    amp_scale = float(getattr(env, "platform_motion_amp_scale", 0.1))
    amp_scale = max(0.0, min(1.0, amp_scale))

    # Convert acceleration limits to amplitude limits for sinusoidal motion.
    omega = 2.0 * math.pi * lin_frequency_hz
    omega_sq = max(omega * omega, 1e-6)
    max_lin_amp = max_linear_acc / omega_sq
    max_ang_amp = max_angular_acc / omega_sq

    lin_amp = max_lin_amp * amp_scale
    ang_amp = max_ang_amp * amp_scale

    # Axis-specific amplitude shaping (x, y, z, roll, pitch, yaw).
    amp_vec = torch.tensor(
        [lin_amp, 0.8 * lin_amp, 0.5 * lin_amp, ang_amp, 0.8 * ang_amp, 0.6 * ang_amp],
        device=env.device,
    )

    motion_mode = str(getattr(env, "platform_motion_mode", "rpy"))
    dof_mask = _platform_motion_dof_mask(motion_mode, level, env.device)

    t = env.episode_length_buf[env_ids_t].float() * env.step_dt
    phase = env._platform_motion_phase[env_ids_t]
    arg = omega * t.unsqueeze(-1) + phase

    sin_part = torch.sin(arg)
    cos_part = torch.cos(arg)
    offsets = sin_part * amp_vec.unsqueeze(0) * dof_mask.unsqueeze(0)

    base_root = env._platform_base_root_state_w[env_ids_t]
    pos_w = base_root[:, :3] + offsets[:, :3]

    quat_delta = quat_from_euler_xyz(offsets[:, 3], offsets[:, 4], offsets[:, 5])
    quat_w = quat_mul(base_root[:, 3:7], quat_delta)

    asset.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=env_ids_t)

    vel = torch.zeros((len(env_ids_t), 6), device=env.device)
    vel[:, :3] = omega * cos_part[:, :3] * amp_vec[:3].unsqueeze(0) * dof_mask[:3].unsqueeze(0)
    vel[:, 3:] = omega * cos_part[:, 3:] * amp_vec[3:].unsqueeze(0) * dof_mask[3:].unsqueeze(0)
    asset.write_root_velocity_to_sim(vel, env_ids=env_ids_t)


def reset_platform_state(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
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

    # 为这些 env 重新采样正弦相位，供后续 episode 使用。
    # 由于 reset 之后 episode_length_buf 会回到 0，平台会先从默认 pose 起步，
    # 下一步再按新的随机相位开始运动。
    env._platform_motion_phase[env_ids_t] = 2.0 * math.pi * torch.rand((len(env_ids_t), 6), device=env.device)


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
