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
