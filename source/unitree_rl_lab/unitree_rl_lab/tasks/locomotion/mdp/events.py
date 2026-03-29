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
        - DoF and amplitude scale are read from environment attributes set by curriculum:
          ``env.platform_motion_level`` in [1, 6] and ``env.platform_motion_amp_scale`` in (0, 1].
    """
    if len(env_ids) == 0:
        return

    asset: RigidObject = env.scene[asset_cfg.name]
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    # Lazy initialization of cached state and per-env phase.
    if not hasattr(env, "_platform_base_root_state_w"):
        default_root = asset.data.default_root_state.clone()
        default_root[:, :3] += env.scene.env_origins
        env._platform_base_root_state_w = default_root
        env._platform_motion_phase = 2.0 * math.pi * torch.rand((env.num_envs, 6), device=env.device)

    level = int(getattr(env, "platform_motion_level", 1))
    level = max(0, min(6, level))
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

    # Enable first N DoFs according to curriculum level.
    dof_mask = torch.zeros(6, device=env.device)
    dof_mask[:level] = 1.0

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
