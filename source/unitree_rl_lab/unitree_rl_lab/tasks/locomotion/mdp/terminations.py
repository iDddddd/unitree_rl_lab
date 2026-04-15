from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import quat_inv, quat_mul

# 该文件定义了与环境终止条件相关的函数，例如判断机器人是否离开平台边界等。这些函数通常会在环境的step函数中被调用，以决定是否需要终止当前episode。
def outside_platform_bounds(
    env: ManagerBasedRLEnv,
    margin: float = 0.2,
    half_size_x: float = 4.0,
    half_size_y: float = 4.0,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Terminate if robot root leaves platform top rectangle in platform frame.

    Args:
        margin: Safety margin from platform edges.
        half_size_x: Platform half-size in x (meters).
        half_size_y: Platform half-size in y (meters).
    """
    robot: Articulation = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]

    # Robot root position in world frame.
    robot_pos_w = robot.data.root_pos_w
    # Platform pose in world frame.
    platform_pos_w = platform.data.root_pos_w
    platform_quat_w = platform.data.root_quat_w

    # Express robot root position in platform frame.
    rel_pos_w = robot_pos_w - platform_pos_w
    rel_pos_p = quat_apply_inverse(platform_quat_w, rel_pos_w)

    limit_x = max(0.0, half_size_x - margin)
    limit_y = max(0.0, half_size_y - margin)

    out_x = torch.abs(rel_pos_p[:, 0]) > limit_x
    out_y = torch.abs(rel_pos_p[:, 1]) > limit_y
    return torch.logical_or(out_x, out_y)


def bad_orientation_rel_platform(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Terminate if robot roll/pitch relative to platform exceeds limit.

    The built-in bad_orientation uses world-frame projected gravity, which
    incorrectly triggers when the platform itself is tilted.  This version
    checks the relative quaternion between robot and platform.
    """
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]

    # Relative quaternion: q_rel = q_platform^{-1} * q_robot  (w,x,y,z)
    quat_rel = quat_mul(quat_inv(platform.data.root_quat_w), robot.data.root_quat_w)
    # Tilt angle from the "upright relative to platform" orientation
    # cos(tilt) = 1 - 2*(x^2 + y^2)  for small angles; use acos for full range
    cos_tilt = 1.0 - 2.0 * (quat_rel[:, 1] ** 2 + quat_rel[:, 2] ** 2)
    tilt_angle = torch.acos(cos_tilt.clamp(-1.0, 1.0))
    return tilt_angle > limit_angle


def root_height_below_minimum_rel_platform(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Terminate if robot height above platform (along platform normal) is below minimum.

    The built-in root_height_below_minimum uses world-frame z, which breaks
    when the platform translates vertically or tilts.
    """
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]

    rel_pos_w = robot.data.root_pos_w - platform.data.root_pos_w
    rel_pos_p = quat_apply_inverse(platform.data.root_quat_w, rel_pos_w)
    # Height along platform normal
    return rel_pos_p[:, 2] < minimum_height
