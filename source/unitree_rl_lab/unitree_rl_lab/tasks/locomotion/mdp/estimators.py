from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

try:
    from isaaclab.utils.math import (
        normalize,
        quat_apply,
        quat_apply_inverse,
        quat_from_angle_axis,
        quat_inv,
        quat_mul,
        quat_unique,
    )
except ImportError:
    from isaaclab.utils.math import normalize, quat_from_angle_axis, quat_inv, quat_mul, quat_unique
    from isaaclab.utils.math import quat_rotate as quat_apply
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, Imu


def _batch_quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply quaternions to vectors with an optional body dimension."""
    original_shape = vec.shape
    if vec.dim() == 3:
        quat = quat[:, None, :].expand(-1, vec.shape[1], -1).reshape(-1, 4)
        vec = vec.reshape(-1, 3)
        return quat_apply(quat, vec).reshape(original_shape)
    return quat_apply(quat, vec)


def _batch_quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply inverse quaternions to vectors with an optional body dimension."""
    original_shape = vec.shape
    if vec.dim() == 3:
        quat = quat[:, None, :].expand(-1, vec.shape[1], -1).reshape(-1, 4)
        vec = vec.reshape(-1, 3)
        return quat_apply_inverse(quat, vec).reshape(original_shape)
    return quat_apply_inverse(quat, vec)


def _integrate_quat_world(quat_w: torch.Tensor, ang_vel_w: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate a world-frame angular velocity into a quaternion."""
    delta_theta = ang_vel_w * dt
    angle = torch.linalg.norm(delta_theta, dim=-1)
    safe_angle = torch.clamp(angle, min=1.0e-9)
    axis = delta_theta / safe_angle.unsqueeze(-1)
    delta_quat = quat_from_angle_axis(angle, axis)
    return quat_unique(normalize(quat_mul(delta_quat, quat_w)))


def _weighted_average_quat(quats: torch.Tensor, weights: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Average a small set of quaternions after aligning their signs."""
    ref = quats[:, 0]
    aligned = quats.clone()
    dots = torch.sum(aligned * ref[:, None, :], dim=-1, keepdim=True)
    aligned = torch.where(dots < 0.0, -aligned, aligned)
    quat = torch.sum(aligned * weights[..., None], dim=1)
    quat_norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    quat = torch.where(quat_norm > 1.0e-6, quat / quat_norm, fallback)
    return quat_unique(normalize(quat))


@dataclass
class PlatformRelativeEKFOutput:
    """Relative robot-platform state produced by :class:`PlatformRelativeEKF`."""

    base_pos_rel_platform: torch.Tensor
    base_vel_rel_platform: torch.Tensor
    base_quat_rel_platform: torch.Tensor
    foot_anchor_pos_platform: torch.Tensor
    foot_pos_rel_platform: torch.Tensor
    contact_prob: torch.Tensor
    platform_quat_w_est: torch.Tensor
    platform_ang_vel_w_est: torch.Tensor
    platform_lin_acc_w_est: torch.Tensor
    measurement_update_mask: torch.Tensor


class PlatformRelativeEKF:
    """Batched relative-state Kalman filter for biped locomotion on an unknown moving platform.

    The filter estimates only the quantities needed by the task: the robot base pose/velocity relative
    to a platform-attached frame plus the two foot anchors on that frame. The platform frame is *not*
    taken from IsaacLab truth. Instead, it is internally propagated from foot-mounted IMUs gated by
    foot contact sensors.

    Current implementation notes:
    - Platform attitude and inertial inputs are non-privileged and come from the fused foot IMUs.
    - Relative base translation/velocity and the stance anchors are estimated with a linear time-varying
      Kalman filter in the platform frame.
    - Robot-side foot kinematics are currently proxied from IsaacLab link states. This matches the
      intended role of encoder/FK/Jacobian terms, but it is still cleaner than using platform truth.
    - If both feet are airborne, the filter only propagates and inflates process noise on both anchors.
    """

    state_dim = 12
    num_feet = 2
    measurement_dim = 12

    def __init__(
        self,
        robot_name: str = "robot",
        base_body_name: str = "torso_link",
        foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
        base_imu_sensor_name: str = "base_imu",
        foot_imu_sensor_names: tuple[str, str] = ("left_foot_imu", "right_foot_imu"),
        foot_contact_sensor_names: tuple[str, str] = ("left_foot_contact", "right_foot_contact"),
        contact_force_threshold: float = 5.0,
        contact_probability_smoothing: float = 0.2,
        stance_probability_threshold: float = 0.5,
        attitude_correction_gain: float = 0.15,
        process_noise_rel_pos: float = 1.0e-3,
        process_noise_rel_vel: float = 4.0e-2,
        process_noise_rel_pos_airborne: float = 1.0e-2,
        process_noise_rel_vel_airborne: float = 4.0e-1,
        process_noise_anchor_stance: float = 1.0e-5,
        process_noise_anchor_swing: float = 5.0e-2,
        process_noise_anchor_airborne: float = 2.5e-1,
        meas_noise_foot_pos_stance: float = 2.0e-4,
        meas_noise_foot_pos_swing: float = 2.0e-1,
        meas_noise_foot_vel_stance: float = 2.0e-3,
        meas_noise_foot_vel_swing: float = 5.0e-1,
        init_covariance: float = 1.0e-2,
        dt: float | None = None,
    ):
        self.robot_name = robot_name
        self.base_body_name = base_body_name
        self.foot_body_names = foot_body_names
        self.base_imu_sensor_name = base_imu_sensor_name
        self.foot_imu_sensor_names = foot_imu_sensor_names
        self.foot_contact_sensor_names = foot_contact_sensor_names

        self.contact_force_threshold = contact_force_threshold
        self.contact_probability_smoothing = contact_probability_smoothing
        self.stance_probability_threshold = stance_probability_threshold
        self.attitude_correction_gain = attitude_correction_gain

        self.process_noise_rel_pos = process_noise_rel_pos
        self.process_noise_rel_vel = process_noise_rel_vel
        self.process_noise_rel_pos_airborne = process_noise_rel_pos_airborne
        self.process_noise_rel_vel_airborne = process_noise_rel_vel_airborne
        self.process_noise_anchor_stance = process_noise_anchor_stance
        self.process_noise_anchor_swing = process_noise_anchor_swing
        self.process_noise_anchor_airborne = process_noise_anchor_airborne

        self.meas_noise_foot_pos_stance = meas_noise_foot_pos_stance
        self.meas_noise_foot_pos_swing = meas_noise_foot_pos_swing
        self.meas_noise_foot_vel_stance = meas_noise_foot_vel_stance
        self.meas_noise_foot_vel_swing = meas_noise_foot_vel_swing

        self.init_covariance = init_covariance
        self.dt = dt

        self._resolved = False
        self._buffers_ready = False

    def reset(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> PlatformRelativeEKFOutput:
        self._ensure_setup(env)
        env_ids = self._canonical_env_ids(env, env_ids)

        kinematics = self._compute_kinematics_proxy(env_ids)
        measured_contact = self._measure_contacts(env_ids)
        self.contact_prob[env_ids] = measured_contact
        self.prev_contact_prob[env_ids] = measured_contact

        fused = self._fuse_platform_imu(env_ids, measured_contact)
        self.platform_quat_w[env_ids] = fused["platform_quat_w"]
        self.platform_ang_vel_w[env_ids] = fused["platform_ang_vel_w"]
        self.platform_ang_acc_w[env_ids] = fused["platform_ang_acc_w"]
        self.platform_lin_acc_w[env_ids] = fused["platform_lin_acc_w"]

        quat_rel = quat_mul(quat_inv(self.platform_quat_w[env_ids]), kinematics["base_quat_w"])
        foot_pos_rel_platform = _batch_quat_apply(quat_rel, kinematics["foot_rel_pos_b"])
        foot_vel_rel_platform = self._compute_contact_point_velocity_platform(quat_rel, kinematics, env_ids)

        base_pos_rel_platform, foot_anchor_pos_platform = self._initialize_relative_state(
            foot_pos_rel_platform,
            measured_contact,
        )
        base_vel_rel_platform = self._initialize_relative_velocity(foot_pos_rel_platform, foot_vel_rel_platform, measured_contact)

        self.x[env_ids, 0:3] = base_pos_rel_platform
        self.x[env_ids, 3:6] = base_vel_rel_platform
        self.x[env_ids, 6:] = foot_anchor_pos_platform.reshape(-1, 6)

        eye = torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
        self.P[env_ids] = eye.unsqueeze(0) * self.init_covariance
        self.measurement_update_mask[env_ids] = measured_contact.max(dim=-1).values > self.stance_probability_threshold
        self.is_initialized[env_ids] = True
        return self._build_output(env_ids, quat_rel)

    def step(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> PlatformRelativeEKFOutput:
        self._ensure_setup(env)
        env_ids = self._canonical_env_ids(env, env_ids)

        not_ready = ~self.is_initialized[env_ids]
        if torch.any(not_ready):
            self.reset(env, env_ids[not_ready])

        dt = self.dt if self.dt is not None else float(env.step_dt)
        if dt <= 0.0:
            raise ValueError(f"PlatformRelativeEKF dt must be positive, got {dt}.")

        measured_contact = self._measure_contacts(env_ids)
        smoothed_contact = (1.0 - self.contact_probability_smoothing) * self.contact_prob[env_ids] + (
            self.contact_probability_smoothing * measured_contact
        )
        self.contact_prob[env_ids] = smoothed_contact
        stance_mask = smoothed_contact > self.stance_probability_threshold
        was_stance_mask = self.prev_contact_prob[env_ids] > self.stance_probability_threshold
        touchdown_mask = stance_mask & (~was_stance_mask)
        airborne_mask = ~torch.any(stance_mask, dim=-1)

        fused = self._fuse_platform_imu(env_ids, smoothed_contact)
        propagated_quat = _integrate_quat_world(self.platform_quat_w[env_ids], fused["platform_ang_vel_w"], dt)
        corrected_quat = _weighted_average_quat(
            torch.stack((propagated_quat, fused["platform_quat_w"]), dim=1),
            torch.stack(
                (
                    torch.full((env_ids.numel(),), 1.0 - self.attitude_correction_gain, device=self.device, dtype=self.dtype),
                    self.attitude_correction_gain * torch.clamp(torch.sum(smoothed_contact, dim=-1), max=1.0),
                ),
                dim=-1,
            ),
            fallback=propagated_quat,
        )
        self.platform_quat_w[env_ids] = corrected_quat
        self.platform_ang_vel_w[env_ids] = fused["platform_ang_vel_w"]
        self.platform_ang_acc_w[env_ids] = fused["platform_ang_acc_w"]
        self.platform_lin_acc_w[env_ids] = fused["platform_lin_acc_w"]

        kinematics = self._compute_kinematics_proxy(env_ids)
        quat_rel = quat_mul(quat_inv(self.platform_quat_w[env_ids]), kinematics["base_quat_w"])
        foot_pos_rel_platform = _batch_quat_apply(quat_rel, kinematics["foot_rel_pos_b"])
        foot_vel_rel_platform = self._compute_contact_point_velocity_platform(quat_rel, kinematics, env_ids)

        x_prev = self.x[env_ids]
        xbar = x_prev.clone()
        rel_pos = x_prev[:, 0:3]
        rel_vel = x_prev[:, 3:6]

        base_lin_acc_platform = _batch_quat_apply(quat_rel, kinematics["base_lin_acc_b"])
        platform_lin_acc_platform = quat_apply_inverse(self.platform_quat_w[env_ids], self.platform_lin_acc_w[env_ids])
        platform_ang_vel_platform = quat_apply_inverse(self.platform_quat_w[env_ids], self.platform_ang_vel_w[env_ids])
        platform_ang_acc_platform = quat_apply_inverse(self.platform_quat_w[env_ids], self.platform_ang_acc_w[env_ids])

        rel_acc_platform = (
            base_lin_acc_platform
            - platform_lin_acc_platform
            - torch.cross(platform_ang_acc_platform, rel_pos, dim=-1)
            - 2.0 * torch.cross(platform_ang_vel_platform, rel_vel, dim=-1)
            - torch.cross(platform_ang_vel_platform, torch.cross(platform_ang_vel_platform, rel_pos, dim=-1), dim=-1)
        )

        xbar[:, 0:3] = rel_pos + dt * rel_vel + 0.5 * dt * dt * rel_acc_platform
        xbar[:, 3:6] = rel_vel + dt * rel_acc_platform

        A = torch.eye(self.state_dim, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(env_ids.numel(), 1, 1)
        A[:, 0:3, 3:6] = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0) * dt
        Pbar = A @ self.P[env_ids] @ A.transpose(1, 2) + self._build_process_noise(smoothed_contact, airborne_mask, dt)

        if torch.any(touchdown_mask):
            xbar = self._reseed_touchdown_anchors(xbar, foot_pos_rel_platform, touchdown_mask)

        H = self._build_measurement_matrix(env_ids.numel())
        yhat = self._predict_measurement(xbar, foot_pos_rel_platform, foot_vel_rel_platform)
        innovation = -yhat
        R = self._build_measurement_noise(smoothed_contact)

        update_mask = torch.any(stance_mask, dim=-1)
        self.measurement_update_mask[env_ids] = update_mask
        if torch.any(update_mask):
            update_ids = torch.nonzero(update_mask, as_tuple=False).squeeze(-1)
            Pbar_sel = Pbar[update_ids]
            H_sel = H[update_ids]
            R_sel = R[update_ids]
            innovation_sel = innovation[update_ids]

            S = H_sel @ Pbar_sel @ H_sel.transpose(1, 2) + R_sel
            S = 0.5 * (S + S.transpose(1, 2))
            PHt = Pbar_sel @ H_sel.transpose(1, 2)
            K = torch.linalg.solve(S, PHt.transpose(1, 2)).transpose(1, 2)

            xbar[update_ids] = xbar[update_ids] + (K @ innovation_sel.unsqueeze(-1)).squeeze(-1)
            Pbar[update_ids] = Pbar_sel - K @ H_sel @ Pbar_sel
            Pbar[update_ids] = 0.5 * (Pbar[update_ids] + Pbar[update_ids].transpose(1, 2))

        self.x[env_ids] = xbar
        self.P[env_ids] = Pbar
        self.prev_contact_prob[env_ids] = smoothed_contact
        return self._build_output(env_ids, quat_rel)

    def _ensure_setup(self, env: ManagerBasedRLEnv) -> None:
        if not self._resolved:
            self._robot: Articulation = env.scene[self.robot_name]
            self._base_imu: Imu = env.scene.sensors[self.base_imu_sensor_name]
            self._foot_imus: tuple[Imu, Imu] = tuple(env.scene.sensors[name] for name in self.foot_imu_sensor_names)
            self._foot_contacts: tuple[ContactSensor, ContactSensor] = tuple(
                env.scene.sensors[name] for name in self.foot_contact_sensor_names
            )
            self._foot_body_ids = torch.as_tensor(
                self._robot.find_bodies(list(self.foot_body_names), preserve_order=True)[0],
                device=env.device,
                dtype=torch.long,
            )
            self._resolved = True

        if not self._buffers_ready:
            self.device = env.device
            self.dtype = self._robot.data.root_pos_w.dtype
            self.x = torch.zeros((env.num_envs, self.state_dim), device=self.device, dtype=self.dtype)
            self.P = torch.zeros((env.num_envs, self.state_dim, self.state_dim), device=self.device, dtype=self.dtype)
            self.platform_quat_w = torch.zeros((env.num_envs, 4), device=self.device, dtype=self.dtype)
            self.platform_quat_w[:, 0] = 1.0
            self.platform_ang_vel_w = torch.zeros((env.num_envs, 3), device=self.device, dtype=self.dtype)
            self.platform_ang_acc_w = torch.zeros((env.num_envs, 3), device=self.device, dtype=self.dtype)
            self.platform_lin_acc_w = torch.zeros((env.num_envs, 3), device=self.device, dtype=self.dtype)
            self.contact_prob = torch.zeros((env.num_envs, self.num_feet), device=self.device, dtype=self.dtype)
            self.prev_contact_prob = torch.zeros((env.num_envs, self.num_feet), device=self.device, dtype=self.dtype)
            self.measurement_update_mask = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
            self.is_initialized = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
            self._buffers_ready = True

    def _canonical_env_ids(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    def _measure_contacts(self, env_ids: torch.Tensor) -> torch.Tensor:
        contacts = []
        for sensor in self._foot_contacts:
            if hasattr(sensor.data, "current_contact_time") and sensor.data.current_contact_time is not None:
                contact_time = sensor.data.current_contact_time[env_ids]
                if contact_time.dim() > 1:
                    contact_time = contact_time.squeeze(-1)
                contact = (contact_time > 0.0).to(dtype=self.dtype)
            else:
                net_force = sensor.data.net_forces_w[env_ids]
                if net_force.dim() > 2:
                    net_force = net_force.squeeze(1)
                contact = (torch.linalg.norm(net_force, dim=-1) > self.contact_force_threshold).to(dtype=self.dtype)
            contacts.append(contact)
        return torch.stack(contacts, dim=-1)

    def _fuse_platform_imu(self, env_ids: torch.Tensor, contact_prob: torch.Tensor) -> dict[str, torch.Tensor]:
        foot_quat_w = []
        foot_ang_vel_w = []
        foot_ang_acc_w = []
        foot_lin_acc_w = []
        for sensor in self._foot_imus:
            quat_w = sensor.data.quat_w[env_ids]
            foot_quat_w.append(quat_w)
            foot_ang_vel_w.append(quat_apply(quat_w, sensor.data.ang_vel_b[env_ids]))
            foot_ang_acc_w.append(quat_apply(quat_w, sensor.data.ang_acc_b[env_ids]))
            foot_lin_acc_w.append(quat_apply(quat_w, sensor.data.lin_acc_b[env_ids]))

        foot_quat_w = torch.stack(foot_quat_w, dim=1)
        foot_ang_vel_w = torch.stack(foot_ang_vel_w, dim=1)
        foot_ang_acc_w = torch.stack(foot_ang_acc_w, dim=1)
        foot_lin_acc_w = torch.stack(foot_lin_acc_w, dim=1)

        weight_sum = torch.sum(contact_prob, dim=-1, keepdim=True)
        safe_weight_sum = torch.clamp(weight_sum, min=1.0)
        weights = contact_prob / safe_weight_sum

        has_contact = weight_sum.squeeze(-1) > 1.0e-6
        fallback_quat = self.platform_quat_w[env_ids]
        fused_quat = _weighted_average_quat(foot_quat_w, weights, fallback=fallback_quat)
        fused_ang_vel_w = torch.sum(foot_ang_vel_w * weights[..., None], dim=1)
        fused_ang_acc_w = torch.sum(foot_ang_acc_w * weights[..., None], dim=1)
        fused_lin_acc_w = torch.sum(foot_lin_acc_w * weights[..., None], dim=1)

        fused_quat = torch.where(has_contact[:, None], fused_quat, fallback_quat)
        fused_ang_vel_w = torch.where(has_contact[:, None], fused_ang_vel_w, self.platform_ang_vel_w[env_ids])
        fused_ang_acc_w = torch.where(has_contact[:, None], fused_ang_acc_w, self.platform_ang_acc_w[env_ids])
        fused_lin_acc_w = torch.where(has_contact[:, None], fused_lin_acc_w, self.platform_lin_acc_w[env_ids])

        return {
            "platform_quat_w": fused_quat,
            "platform_ang_vel_w": fused_ang_vel_w,
            "platform_ang_acc_w": fused_ang_acc_w,
            "platform_lin_acc_w": fused_lin_acc_w,
        }

    def _compute_kinematics_proxy(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        base_quat_w = self._base_imu.data.quat_w[env_ids]
        base_pos_w = self._base_imu.data.pos_w[env_ids]
        base_lin_vel_w = quat_apply(base_quat_w, self._base_imu.data.lin_vel_b[env_ids])
        base_lin_acc_b = self._base_imu.data.lin_acc_b[env_ids]
        base_ang_vel_b = self._base_imu.data.ang_vel_b[env_ids]

        foot_pos_w = self._robot.data.body_pos_w[env_ids][:, self._foot_body_ids]
        foot_vel_w = self._robot.data.body_lin_vel_w[env_ids][:, self._foot_body_ids]

        foot_rel_pos_w = foot_pos_w - base_pos_w[:, None, :]
        foot_rel_pos_b = _batch_quat_apply_inverse(base_quat_w, foot_rel_pos_w)

        foot_rel_vel_w = foot_vel_w - base_lin_vel_w[:, None, :]
        foot_rel_vel_b = _batch_quat_apply_inverse(base_quat_w, foot_rel_vel_w)
        foot_rel_pos_dot_b = foot_rel_vel_b - torch.cross(
            base_ang_vel_b[:, None, :].expand_as(foot_rel_pos_b),
            foot_rel_pos_b,
            dim=-1,
        )

        return {
            "base_quat_w": base_quat_w,
            "base_lin_acc_b": base_lin_acc_b,
            "base_ang_vel_b": base_ang_vel_b,
            "foot_rel_pos_b": foot_rel_pos_b,
            "foot_rel_pos_dot_b": foot_rel_pos_dot_b,
        }

    def _compute_contact_point_velocity_platform(
        self,
        quat_rel: torch.Tensor,
        kinematics: dict[str, torch.Tensor],
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        foot_pos_rel_platform = _batch_quat_apply(quat_rel, kinematics["foot_rel_pos_b"])
        foot_pos_dot_platform = _batch_quat_apply(quat_rel, kinematics["foot_rel_pos_dot_b"])
        base_ang_vel_platform = quat_apply(quat_rel, kinematics["base_ang_vel_b"])
        platform_ang_vel_platform = quat_apply_inverse(self.platform_quat_w[env_ids], self.platform_ang_vel_w[env_ids])
        omega_rel_platform = base_ang_vel_platform - platform_ang_vel_platform
        return foot_pos_dot_platform + torch.cross(
            omega_rel_platform[:, None, :].expand_as(foot_pos_rel_platform),
            foot_pos_rel_platform,
            dim=-1,
        )

    def _initialize_relative_state(
        self,
        foot_pos_rel_platform: torch.Tensor,
        contact_prob: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = foot_pos_rel_platform.shape[0]
        base_pos_rel_platform = torch.zeros((batch, 3), device=self.device, dtype=self.dtype)
        foot_anchor_pos_platform = torch.zeros((batch, self.num_feet, 3), device=self.device, dtype=self.dtype)
        stance_mask = contact_prob > self.stance_probability_threshold

        both_stance = torch.all(stance_mask, dim=-1)
        left_only = stance_mask[:, 0] & (~stance_mask[:, 1])
        right_only = stance_mask[:, 1] & (~stance_mask[:, 0])
        none_stance = ~torch.any(stance_mask, dim=-1)

        if torch.any(both_stance):
            midpoint = 0.5 * (foot_pos_rel_platform[both_stance, 0] + foot_pos_rel_platform[both_stance, 1])
            base_pos_rel_platform[both_stance] = -midpoint
        if torch.any(left_only):
            base_pos_rel_platform[left_only] = -foot_pos_rel_platform[left_only, 0]
        if torch.any(right_only):
            base_pos_rel_platform[right_only] = -foot_pos_rel_platform[right_only, 1]
        if torch.any(none_stance):
            midpoint = torch.mean(foot_pos_rel_platform[none_stance], dim=1)
            base_pos_rel_platform[none_stance] = -midpoint

        foot_anchor_pos_platform = base_pos_rel_platform[:, None, :] + foot_pos_rel_platform
        return base_pos_rel_platform, foot_anchor_pos_platform

    def _initialize_relative_velocity(
        self,
        foot_pos_rel_platform: torch.Tensor,
        foot_vel_rel_platform: torch.Tensor,
        contact_prob: torch.Tensor,
    ) -> torch.Tensor:
        del foot_pos_rel_platform
        batch = foot_vel_rel_platform.shape[0]
        base_vel_rel_platform = torch.zeros((batch, 3), device=self.device, dtype=self.dtype)
        stance_mask = contact_prob > self.stance_probability_threshold
        valid = torch.any(stance_mask, dim=-1)
        if torch.any(valid):
            masked_vel = torch.where(stance_mask[..., None], -foot_vel_rel_platform, 0.0)
            denom = torch.clamp(torch.sum(stance_mask.to(self.dtype), dim=-1, keepdim=True), min=1.0)
            base_vel_rel_platform[valid] = torch.sum(masked_vel[valid], dim=1) / denom[valid]
        return base_vel_rel_platform

    def _reseed_touchdown_anchors(
        self,
        state: torch.Tensor,
        foot_pos_rel_platform: torch.Tensor,
        touchdown_mask: torch.Tensor,
    ) -> torch.Tensor:
        state = state.clone()
        foot_anchor_pos_platform = state[:, 6:].reshape(-1, self.num_feet, 3)
        current_foot_platform = state[:, None, 0:3] + foot_pos_rel_platform
        foot_anchor_pos_platform = torch.where(
            touchdown_mask[..., None],
            current_foot_platform,
            foot_anchor_pos_platform,
        )
        state[:, 6:] = foot_anchor_pos_platform.reshape(-1, 6)
        return state

    def _build_process_noise(self, contact_prob: torch.Tensor, airborne_mask: torch.Tensor, dt: float) -> torch.Tensor:
        batch = contact_prob.shape[0]
        Q = torch.zeros((batch, self.state_dim, self.state_dim), device=self.device, dtype=self.dtype)
        eye3 = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)

        rel_pos_noise = torch.where(
            airborne_mask,
            torch.full((batch,), self.process_noise_rel_pos_airborne, device=self.device, dtype=self.dtype),
            torch.full((batch,), self.process_noise_rel_pos, device=self.device, dtype=self.dtype),
        )
        rel_vel_noise = torch.where(
            airborne_mask,
            torch.full((batch,), self.process_noise_rel_vel_airborne, device=self.device, dtype=self.dtype),
            torch.full((batch,), self.process_noise_rel_vel, device=self.device, dtype=self.dtype),
        )

        Q[:, 0:3, 0:3] = eye3 * rel_pos_noise.view(-1, 1, 1) * dt * dt
        Q[:, 3:6, 3:6] = eye3 * rel_vel_noise.view(-1, 1, 1) * dt

        anchor_noise = self.process_noise_anchor_stance * contact_prob + self.process_noise_anchor_swing * (1.0 - contact_prob)
        anchor_noise = torch.where(
            airborne_mask[:, None],
            torch.full_like(anchor_noise, self.process_noise_anchor_airborne),
            anchor_noise,
        )
        Q[:, 6:9, 6:9] = eye3 * anchor_noise[:, 0].view(-1, 1, 1)
        Q[:, 9:12, 9:12] = eye3 * anchor_noise[:, 1].view(-1, 1, 1)
        return Q

    def _build_measurement_matrix(self, batch: int) -> torch.Tensor:
        H = torch.zeros((batch, self.measurement_dim, self.state_dim), device=self.device, dtype=self.dtype)
        eye3 = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)

        H[:, 0:3, 0:3] = eye3
        H[:, 0:3, 6:9] = -eye3
        H[:, 3:6, 3:6] = eye3

        H[:, 6:9, 0:3] = eye3
        H[:, 6:9, 9:12] = -eye3
        H[:, 9:12, 3:6] = eye3
        return H

    def _predict_measurement(
        self,
        state: torch.Tensor,
        foot_pos_rel_platform: torch.Tensor,
        foot_vel_rel_platform: torch.Tensor,
    ) -> torch.Tensor:
        yhat = torch.zeros((state.shape[0], self.measurement_dim), device=self.device, dtype=self.dtype)
        foot_anchor_pos_platform = state[:, 6:].reshape(-1, self.num_feet, 3)
        base_pos_rel_platform = state[:, None, 0:3]
        base_vel_rel_platform = state[:, None, 3:6]

        yhat[:, 0:3] = base_pos_rel_platform[:, 0] + foot_pos_rel_platform[:, 0] - foot_anchor_pos_platform[:, 0]
        yhat[:, 3:6] = base_vel_rel_platform[:, 0] + foot_vel_rel_platform[:, 0]
        yhat[:, 6:9] = base_pos_rel_platform[:, 0] + foot_pos_rel_platform[:, 1] - foot_anchor_pos_platform[:, 1]
        yhat[:, 9:12] = base_vel_rel_platform[:, 0] + foot_vel_rel_platform[:, 1]
        return yhat

    def _build_measurement_noise(self, contact_prob: torch.Tensor) -> torch.Tensor:
        batch = contact_prob.shape[0]
        R = torch.zeros((batch, self.measurement_dim, self.measurement_dim), device=self.device, dtype=self.dtype)
        eye3 = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)

        pos_noise = self.meas_noise_foot_pos_stance * contact_prob + self.meas_noise_foot_pos_swing * (1.0 - contact_prob)
        vel_noise = self.meas_noise_foot_vel_stance * contact_prob + self.meas_noise_foot_vel_swing * (1.0 - contact_prob)

        R[:, 0:3, 0:3] = eye3 * pos_noise[:, 0].view(-1, 1, 1)
        R[:, 3:6, 3:6] = eye3 * vel_noise[:, 0].view(-1, 1, 1)
        R[:, 6:9, 6:9] = eye3 * pos_noise[:, 1].view(-1, 1, 1)
        R[:, 9:12, 9:12] = eye3 * vel_noise[:, 1].view(-1, 1, 1)
        return R

    def _build_output(self, env_ids: torch.Tensor, quat_rel: torch.Tensor | None = None) -> PlatformRelativeEKFOutput:
        if quat_rel is None:
            quat_rel = quat_mul(quat_inv(self.platform_quat_w[env_ids]), self._base_imu.data.quat_w[env_ids])

        kinematics = self._compute_kinematics_proxy(env_ids)
        foot_pos_rel_platform = self.x[env_ids, None, 0:3] + _batch_quat_apply(quat_rel, kinematics["foot_rel_pos_b"])

        return PlatformRelativeEKFOutput(
            base_pos_rel_platform=self.x[env_ids, 0:3],
            base_vel_rel_platform=self.x[env_ids, 3:6],
            base_quat_rel_platform=quat_rel,
            foot_anchor_pos_platform=self.x[env_ids, 6:].reshape(-1, self.num_feet, 3),
            foot_pos_rel_platform=foot_pos_rel_platform,
            contact_prob=self.contact_prob[env_ids],
            platform_quat_w_est=self.platform_quat_w[env_ids],
            platform_ang_vel_w_est=self.platform_ang_vel_w[env_ids],
            platform_lin_acc_w_est=self.platform_lin_acc_w[env_ids],
            measurement_update_mask=self.measurement_update_mask[env_ids],
        )


# Backward-compatible alias for older experiments that imported the previous name.
PlatformBipedEKF = PlatformRelativeEKF
PlatformBipedEKFOutput = PlatformRelativeEKFOutput
