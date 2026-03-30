from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def platform_body_vel_deltas_b(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Relative platform-body velocity feature in robot body frame.

    Output is:
        [v_plf_xy^B - v_body_xy^B, w_plf_z^B - w_body_z^B]
    """
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]

    rel_lin_vel_w = platform.data.root_lin_vel_w - robot.data.root_lin_vel_w
    rel_ang_vel_w = platform.data.root_ang_vel_w - robot.data.root_ang_vel_w

    rel_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, rel_lin_vel_w)
    rel_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, rel_ang_vel_w)

    return torch.cat([rel_lin_vel_b[:, :2], rel_ang_vel_b[:, 2:3]], dim=-1)
