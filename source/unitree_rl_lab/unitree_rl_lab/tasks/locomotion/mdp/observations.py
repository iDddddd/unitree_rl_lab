from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse as quat_apply_inverse

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

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


def base_lin_acc_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Base linear acceleration in the robot body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_lin_acc_w = asset.data.body_lin_acc_w[:, asset_cfg.body_ids[0], :]
    return quat_apply_inverse(asset.data.root_quat_w, body_lin_acc_w)


def base_roll_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot roll and pitch angles in radians."""
    asset: RigidObject = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.stack([roll, pitch], dim=-1)


def last_action_subset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    action_name: str | None = None,
) -> torch.Tensor:
    """Select the subset of the last action matching the provided joints."""
    if action_name is None:
        actions = env.action_manager.action
    else:
        actions = env.action_manager.get_term(action_name).raw_actions
    if asset_cfg.joint_ids == slice(None):
        return actions
    return actions[:, asset_cfg.joint_ids]


def binary_foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 10.0,
) -> torch.Tensor:
    """Binary foot contact labels from contact forces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.linalg.norm(net_forces, dim=-1) > threshold
    return contact.float()


def base_lin_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot base linear velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def platform_lin_vel_b(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Platform linear velocity expressed in the robot body frame."""
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, platform.data.root_lin_vel_w)


def platform_ang_vel_b(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Platform angular velocity expressed in the robot body frame."""
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, platform.data.root_ang_vel_w)
