from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul
except ImportError:
    from isaaclab.utils.math import quat_rotate as quat_apply
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
    from isaaclab.utils.math import quat_inv, quat_mul

from isaaclab.assets import RigidObject
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.managers import SceneEntityCfg

from .estimators import PlatformRelativeEKF, PlatformRelativeEKFOutput

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_EKF_DEBUG_MARKERS_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/EKF/base_compare",
    markers={
        "estimate": sim_utils.SphereCfg(
            radius=0.045,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
        ),
        "truth": sim_utils.SphereCfg(
            radius=0.045,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
        ),
    },
)


def _batch_quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply quaternions to a batched tensor with an optional body dimension."""
    original_shape = vec.shape
    if vec.dim() == 3:
        quat = quat[:, None, :].expand(-1, vec.shape[1], -1).reshape(-1, 4)
        vec = vec.reshape(-1, 3)
        return quat_apply(quat, vec).reshape(original_shape)
    return quat_apply(quat, vec)


def _batch_quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply inverse quaternions to a batched tensor with an optional body dimension."""
    original_shape = vec.shape
    if vec.dim() == 3:
        quat = quat[:, None, :].expand(-1, vec.shape[1], -1).reshape(-1, 4)
        vec = vec.reshape(-1, 3)
        return quat_apply_inverse(quat, vec).reshape(original_shape)
    return quat_apply_inverse(quat, vec)


def _quat_angle_error(q_est: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
    """Return the angular distance between two batched quaternions in radians."""
    q_err = quat_mul(quat_inv(q_true), q_est)
    q_err = q_err / torch.linalg.norm(q_err, dim=-1, keepdim=True).clamp_min(1.0e-9)
    imag_norm = torch.linalg.norm(q_err[:, 1:], dim=-1)
    return 2.0 * torch.atan2(imag_norm, torch.abs(q_err[:, 0]))


def _get_platform_relative_ekf_output(env: ManagerBasedRLEnv) -> PlatformRelativeEKFOutput:
    """Compute or fetch the cached EKF output for the current env step."""
    cache = getattr(env, "_platform_relative_ekf_cache", None)
    if cache is None:
        cache = {
            "estimator": PlatformRelativeEKF(),
            "last_step": None,
            "output": None,
            "debug_origin_offset": torch.zeros((env.num_envs, 3), device=env.device),
            "visualizer": None,
        }
        setattr(env, "_platform_relative_ekf_cache", cache)

    current_step = int(getattr(env, "common_step_counter", 0))
    if cache["last_step"] == current_step and cache["output"] is not None:
        return cache["output"]

    all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    reset_ids = torch.nonzero(env.episode_length_buf == 0, as_tuple=False).squeeze(-1)
    if cache["output"] is None:
        reset_ids = all_env_ids

    if reset_ids.numel() > 0:
        cache["estimator"].reset(env, reset_ids)

    step_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    if reset_ids.numel() > 0:
        step_mask[reset_ids] = False
    step_ids = torch.nonzero(step_mask, as_tuple=False).squeeze(-1)
    if step_ids.numel() > 0:
        cache["estimator"].step(env, step_ids)

    output = cache["estimator"]._build_output(all_env_ids)
    cache["output"] = output
    cache["last_step"] = current_step

    if getattr(env.cfg, "ekf_debug_vis", False):
        _update_platform_relative_ekf_debug(env, cache, output, reset_ids)

    return output


def _update_platform_relative_ekf_debug(
    env: ManagerBasedRLEnv,
    cache: dict,
    output: PlatformRelativeEKFOutput,
    reset_ids: torch.Tensor,
) -> None:
    """Log EKF-vs-truth errors and visualize the first play environment."""
    robot: RigidObject = env.scene["robot"]
    platform: RigidObject = env.scene["platform"]

    if "log" not in env.extras:
        env.extras["log"] = {}

    platform_quat_est = output.platform_quat_w_est
    base_pos_w = robot.data.root_pos_w
    base_vel_w = robot.data.root_lin_vel_w
    base_quat_w = robot.data.root_quat_w
    platform_pos_w = platform.data.root_pos_w
    platform_vel_w = platform.data.root_lin_vel_w
    platform_quat_w = platform.data.root_quat_w
    platform_ang_vel_w = platform.data.root_ang_vel_w
    platform_lin_acc_w = platform.data.body_lin_acc_w[:, 0]

    if reset_ids.numel() > 0:
        base_pos_rel_truth_reset = quat_apply_inverse(
            platform_quat_est[reset_ids],
            base_pos_w[reset_ids] - platform_pos_w[reset_ids],
        )
        cache["debug_origin_offset"][reset_ids] = output.base_pos_rel_platform[reset_ids] - base_pos_rel_truth_reset

    base_pos_rel_truth = quat_apply_inverse(platform_quat_est, base_pos_w - platform_pos_w) + cache["debug_origin_offset"]
    base_vel_rel_truth = quat_apply_inverse(
        platform_quat_est,
        base_vel_w - platform_vel_w - torch.cross(platform_ang_vel_w, base_pos_w - platform_pos_w, dim=-1),
    )
    base_quat_rel_truth = quat_mul(quat_inv(platform_quat_w), base_quat_w)

    pos_error = torch.linalg.norm(output.base_pos_rel_platform - base_pos_rel_truth, dim=-1)
    vel_error = torch.linalg.norm(output.base_vel_rel_platform - base_vel_rel_truth, dim=-1)
    quat_error = _quat_angle_error(output.base_quat_rel_platform, base_quat_rel_truth)
    platform_quat_error = _quat_angle_error(platform_quat_est, platform_quat_w)
    platform_ang_vel_error = torch.linalg.norm(output.platform_ang_vel_w_est - platform_ang_vel_w, dim=-1)
    platform_lin_acc_error = torch.linalg.norm(output.platform_lin_acc_w_est - platform_lin_acc_w, dim=-1)

    env.extras["log"]["EKF/base_pos_rel_error"] = pos_error.mean()
    env.extras["log"]["EKF/base_vel_rel_error"] = vel_error.mean()
    env.extras["log"]["EKF/base_quat_rel_error_rad"] = quat_error.mean()
    env.extras["log"]["EKF/platform_quat_error_rad"] = platform_quat_error.mean()
    env.extras["log"]["EKF/platform_ang_vel_error"] = platform_ang_vel_error.mean()
    env.extras["log"]["EKF/platform_lin_acc_error"] = platform_lin_acc_error.mean()

    env_id = int(getattr(env.cfg, "ekf_debug_env_id", 0))
    env_id = max(0, min(env.num_envs - 1, env_id))
    env.extras["ekf_panel"] = {
        "env_id": env_id,
        "base_pos_rel_est": output.base_pos_rel_platform[env_id].detach().cpu().clone(),
        "base_pos_rel_truth": base_pos_rel_truth[env_id].detach().cpu().clone(),
        "base_vel_rel_est": output.base_vel_rel_platform[env_id].detach().cpu().clone(),
        "base_vel_rel_truth": base_vel_rel_truth[env_id].detach().cpu().clone(),
        "base_quat_rel_est": output.base_quat_rel_platform[env_id].detach().cpu().clone(),
        "base_quat_rel_truth": base_quat_rel_truth[env_id].detach().cpu().clone(),
        "base_pos_rel_error": pos_error[env_id].detach().cpu().clone(),
        "base_vel_rel_error": vel_error[env_id].detach().cpu().clone(),
        "base_quat_rel_error_rad": quat_error[env_id].detach().cpu().clone(),
        "platform_quat_error_rad": platform_quat_error[env_id].detach().cpu().clone(),
        "platform_ang_vel_error": platform_ang_vel_error[env_id].detach().cpu().clone(),
        "platform_lin_acc_error": platform_lin_acc_error[env_id].detach().cpu().clone(),
    }

    if cache["visualizer"] is None:
        cache["visualizer"] = VisualizationMarkers(_EKF_DEBUG_MARKERS_CFG)

    base_pos_est_w = platform_pos_w + quat_apply(platform_quat_est, output.base_pos_rel_platform - cache["debug_origin_offset"])
    marker_positions = torch.stack((base_pos_est_w[env_id], base_pos_w[env_id]), dim=0)
    marker_indices = torch.tensor([0, 1], device=env.device, dtype=torch.int32)
    cache["visualizer"].visualize(translations=marker_positions, marker_indices=marker_indices)


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


def ekf_base_pos_rel_platform(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Estimated robot base position relative to the platform-attached EKF frame."""
    return torch.nan_to_num(_get_platform_relative_ekf_output(env).base_pos_rel_platform, nan=0.0, posinf=0.0, neginf=0.0)


def ekf_base_vel_rel_platform(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Estimated robot base velocity relative to the platform-attached EKF frame."""
    return torch.nan_to_num(_get_platform_relative_ekf_output(env).base_vel_rel_platform, nan=0.0, posinf=0.0, neginf=0.0)


def ekf_base_quat_rel_platform(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Estimated robot base orientation relative to the platform-attached EKF frame."""
    return torch.nan_to_num(_get_platform_relative_ekf_output(env).base_quat_rel_platform, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Privileged ground-truth variants (for critic only)
# ---------------------------------------------------------------------------

def gt_base_pos_rel_platform(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Privileged: true robot base position relative to platform, expressed in platform frame."""
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]
    r_rel_w = robot.data.root_pos_w - platform.data.root_pos_w
    pos_rel = quat_apply_inverse(platform.data.root_quat_w, r_rel_w)
    return torch.nan_to_num(pos_rel, nan=0.0, posinf=0.0, neginf=0.0)


def gt_base_vel_rel_platform(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Privileged: true robot base velocity relative to platform, expressed in platform frame."""
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]
    r_rel_w = robot.data.root_pos_w - platform.data.root_pos_w
    # Transport theorem: v_rel = v_base - v_platform - omega_platform × r_rel
    v_rel_w = (
        robot.data.root_lin_vel_w
        - platform.data.root_lin_vel_w
        - torch.cross(platform.data.root_ang_vel_w, r_rel_w, dim=-1)
    )
    vel_rel = quat_apply_inverse(platform.data.root_quat_w, v_rel_w)
    return torch.nan_to_num(vel_rel, nan=0.0, posinf=0.0, neginf=0.0)


def gt_base_quat_rel_platform(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Privileged: true robot base orientation relative to platform."""
    robot: RigidObject = env.scene[robot_asset_cfg.name]
    platform: RigidObject = env.scene[platform_asset_cfg.name]
    quat_rel = quat_mul(quat_inv(platform.data.root_quat_w), robot.data.root_quat_w)
    return torch.nan_to_num(quat_rel, nan=0.0, posinf=0.0, neginf=0.0)
