from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.utils.math import quat_inv, quat_mul
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_relative_to_platform_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Penalize base height error relative to platform height (world-z difference)."""
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]
    rel_height = robot.data.root_pos_w[:, 2] - platform.data.root_pos_w[:, 2]
    return torch.square(rel_height - target_height)


def base_height_relative_to_platform_normal_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Penalize base height error along platform normal direction."""
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]

    rel_pos_w = robot.data.root_pos_w - platform.data.root_pos_w
    rel_pos_p = quat_apply_inverse(platform.data.root_quat_w, rel_pos_w)
    rel_height = rel_pos_p[:, 2]
    return torch.square(rel_height - target_height)


def base_lin_vel_z_rel_platform_l2(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Penalize base z linear velocity relative to platform z linear velocity."""
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]
    rel_lin_vel_z = robot.data.root_lin_vel_w[:, 2] - platform.data.root_lin_vel_w[:, 2]
    return torch.square(rel_lin_vel_z)


def base_ang_vel_xy_rel_platform_l2(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Penalize base xy angular velocity relative to platform xy angular velocity."""
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]
    rel_ang_vel_xy = robot.data.root_ang_vel_w[:, :2] - platform.data.root_ang_vel_w[:, :2]
    return torch.sum(torch.square(rel_ang_vel_xy), dim=1)


def base_orientation_rel_platform_l2(
    env: ManagerBasedRLEnv,
    deadband_rad: float = 0.08,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Penalize robot roll/pitch mismatch with platform using relative quaternion."""
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]

    quat_rel = quat_mul(quat_inv(platform.data.root_quat_w), robot.data.root_quat_w)
    # Quaternion is assumed as [w, x, y, z]. We keep roll/pitch mismatch and ignore yaw.
    tilt_mag = torch.sqrt(torch.clamp(torch.square(quat_rel[:, 1]) + torch.square(quat_rel[:, 2]), min=0.0))
    tilt_err = torch.clamp(tilt_mag - deadband_rad, min=0.0)
    return torch.square(tilt_err)


def platform_body_vel_deltas_l2(
    env: ManagerBasedRLEnv,
    lin_xy_weight: float = 1.0,
    ang_z_weight: float = 0.5,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Penalize relative platform-body velocity mismatch in robot body frame."""
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]

    rel_lin_vel_w = platform.data.root_lin_vel_w - robot.data.root_lin_vel_w
    rel_ang_vel_w = platform.data.root_ang_vel_w - robot.data.root_ang_vel_w
    rel_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, rel_lin_vel_w)
    rel_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, rel_ang_vel_w)

    lin_xy_term = torch.sum(torch.square(rel_lin_vel_b[:, :2]), dim=1)
    ang_z_term = torch.square(rel_ang_vel_b[:, 2])
    return lin_xy_weight * lin_xy_term + ang_z_weight * ang_z_term


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def action_rate_l2_bounded(
    env: ManagerBasedRLEnv,
    action_clip: float = 10.0,
    reward_clip: float = 1.0e3,
) -> torch.Tensor:
    """Penalize action-rate with finite bounds to avoid one bad step poisoning PPO."""
    action = torch.nan_to_num(env.action_manager.action, nan=0.0, posinf=action_clip, neginf=-action_clip)
    prev_action = torch.nan_to_num(env.action_manager.prev_action, nan=0.0, posinf=action_clip, neginf=-action_clip)
    delta = (action - prev_action).clamp(min=-action_clip, max=action_clip)
    return torch.sum(torch.square(delta), dim=1).clamp(max=reward_clip)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_clearance_relative_platform_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    platform_asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward foot clearance measured in platform frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]

    rel_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - platform.data.root_pos_w.unsqueeze(1)
    rel_pos_p = torch.zeros_like(rel_pos_w)
    for i in range(rel_pos_w.shape[1]):
        rel_pos_p[:, i, :] = quat_apply_inverse(platform.data.root_quat_w, rel_pos_w[:, i, :])

    rel_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - platform.data.root_lin_vel_w.unsqueeze(1)
    rel_vel_p = torch.zeros_like(rel_vel_w)
    for i in range(rel_vel_w.shape[1]):
        rel_vel_p[:, i, :] = quat_apply_inverse(platform.data.root_quat_w, rel_vel_w[:, i, :])

    foot_z_target_error = torch.square(rel_pos_p[:, :, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(rel_vel_p[:, :, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])
    reward /= max(1, len(sensor_cfg.body_ids))

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        moving_mask = cmd_norm > 0.05
        reward *= moving_mask

        # Extra anti-shuffle shaping for biped gait.
        if len(sensor_cfg.body_ids) == 2:
            move_scale = torch.clamp((cmd_norm - 0.05) / 0.35, 0.0, 1.0)
            both_contact = (is_contact[:, 0] & is_contact[:, 1]).float()
            no_contact = (~(is_contact[:, 0] | is_contact[:, 1])).float()
            alternating = (is_contact[:, 0] ^ is_contact[:, 1]).float()
            reward = reward + 0.15 * move_scale * alternating - 0.35 * move_scale * both_contact - 0.10 * move_scale * no_contact
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
