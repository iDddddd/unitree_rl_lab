# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--platform_motion_mode",
    type=str,
    default=None,
    choices=["rpy", "xyz", "z_rp", "full"],
    help="Optional fixed platform motion mode for play mode.",
)
parser.add_argument(
    "--platform_motion_level",
    type=int,
    default=None,
    choices=[1, 2, 3, 4, 5, 6],
    help="Optional fixed platform motion level for play mode (1-6).",
)
parser.add_argument(
    "--platform_amp_scale",
    type=float,
    default=None,
    help="Optional fixed platform motion amplitude scale for play mode (range: 0-1.0).",
)
parser.add_argument(
    "--ekf_panel",
    action="store_true",
    default=False,
    help="Open a separate realtime matplotlib panel for EKF-vs-truth curves during play.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import time
import torch
from collections import deque

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.tasks.locomotion.mdp.events import get_platform_motion_mode_max_level

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class EkfRealtimePlotter:
    """Realtime panel for comparing EKF estimates against simulator truth."""

    @staticmethod
    def _quat_to_rpy(q: torch.Tensor) -> tuple[float, float, float]:
        """Convert a (4,) quaternion [w, x, y, z] tensor to roll/pitch/yaw (radians)."""
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(sinp)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return roll, pitch, yaw

    def __init__(self, max_points: int = 200, update_every: int = 2) -> None:
        self.max_points = max_points
        self.update_every = update_every
        self.step = 0
        self.enabled = plt is not None
        self._history = {
            "t": deque(maxlen=max_points),
            "quat_est": [deque(maxlen=max_points) for _ in range(3)],
            "quat_true": [deque(maxlen=max_points) for _ in range(3)],
            "vel_est": [deque(maxlen=max_points) for _ in range(3)],
            "vel_true": [deque(maxlen=max_points) for _ in range(3)],
            "vel_kin_z": deque(maxlen=max_points),
            "vel_err": deque(maxlen=max_points),
            "quat_err": deque(maxlen=max_points),
            "left_contact": deque(maxlen=max_points),
            "right_contact": deque(maxlen=max_points),
        }
        self._fig = None
        self._axes = None
        self._lines = {}
        self._rpy_indices = (0, 1)
        self._vel_indices = (2,)

        if not self.enabled:
            return

        plt.ion()
        self._fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        self._axes = axes
        self._fig.canvas.manager.set_window_title("EKF Realtime Panel")

        colors = ["tab:red", "tab:green", "tab:blue"]
        labels = ["x", "y", "z"]
        rpy_labels = ["roll", "pitch", "yaw"]

        for i in self._rpy_indices:
            (line_quat_est,) = axes[0].plot([], [], color=colors[i], linestyle="-", label=f"{rpy_labels[i]} est")
            (line_quat_true,) = axes[0].plot([], [], color=colors[i], linestyle="--", label=f"{rpy_labels[i]} true")
            self._lines[f"quat_est_{i}"] = line_quat_est
            self._lines[f"quat_true_{i}"] = line_quat_true

        for i in self._vel_indices:
            (line_vel_est,) = axes[1].plot([], [], color=colors[i], linestyle="-", label=f"vel est {labels[i]}")
            (line_vel_true,) = axes[1].plot([], [], color=colors[i], linestyle="--", label=f"vel true {labels[i]}")
            self._lines[f"vel_est_{i}"] = line_vel_est
            self._lines[f"vel_true_{i}"] = line_vel_true
        (line_vel_kin_z,) = axes[1].plot([], [], color="tab:orange", linestyle=":", label="vel kin z")
        self._lines["vel_kin_z"] = line_vel_kin_z

        (line_vel_err,) = axes[2].plot([], [], color="tab:purple", label="vel true z - est z")
        (line_quat_err,) = axes[2].plot([], [], color="tab:brown", label="roll/pitch error rad")
        self._lines["vel_err"] = line_vel_err
        self._lines["quat_err"] = line_quat_err

        self._contact_text = axes[2].text(
            0.01,
            0.95,
            "",
            transform=axes[2].transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "edgecolor": "0.75"},
        )

        axes[0].set_ylabel("Rel Orientation (rad)")
        axes[1].set_ylabel("Base Rel Vel")
        axes[2].set_ylabel("Errors")
        axes[2].set_xlabel("Step")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", ncol=2, fontsize=8)
        axes[1].legend(loc="upper right", ncol=2, fontsize=8)
        axes[2].legend(loc="upper right", fontsize=8)
        self._fig.tight_layout()

    def update(self, extras: dict, step_idx: int) -> None:
        if not self.enabled or self._fig is None or extras is None:
            return
        if not plt.fignum_exists(self._fig.number):
            self.enabled = False
            return

        panel_data = extras.get("ekf_panel")
        if not panel_data:
            return

        self._history["t"].append(step_idx)
        rpy_est = self._quat_to_rpy(panel_data["base_quat_rel_est"])
        rpy_true = self._quat_to_rpy(panel_data["base_quat_rel_truth"])
        for i in self._rpy_indices:
            self._history["quat_est"][i].append(rpy_est[i])
            self._history["quat_true"][i].append(rpy_true[i])
        for i in self._vel_indices:
            self._history["vel_est"][i].append(float(panel_data["base_vel_rel_est"][i]))
            self._history["vel_true"][i].append(float(panel_data["base_vel_rel_truth"][i]))

        self._history["vel_kin_z"].append(float(panel_data.get("base_vel_rel_kin_z", float("nan"))))
        vel_z_error = float(panel_data["base_vel_rel_truth"][2]) - float(panel_data["base_vel_rel_est"][2])
        self._history["vel_err"].append(vel_z_error)
        self._history["quat_err"].append(float(panel_data["base_quat_rel_error_rad"]))
        contact_prob = torch.as_tensor(panel_data.get("foot_contact_prob", [float("nan"), float("nan")]))
        left_prob = float(contact_prob[0]) if contact_prob.numel() > 0 else float("nan")
        right_prob = float(contact_prob[1]) if contact_prob.numel() > 1 else float("nan")
        self._history["left_contact"].append(left_prob)
        self._history["right_contact"].append(right_prob)

        self.step += 1
        if self.step % self.update_every != 0:
            return

        t = list(self._history["t"])
        for i in self._rpy_indices:
            self._lines[f"quat_est_{i}"].set_data(t, list(self._history["quat_est"][i]))
            self._lines[f"quat_true_{i}"].set_data(t, list(self._history["quat_true"][i]))
        for i in self._vel_indices:
            self._lines[f"vel_est_{i}"].set_data(t, list(self._history["vel_est"][i]))
            self._lines[f"vel_true_{i}"].set_data(t, list(self._history["vel_true"][i]))

        self._lines["vel_err"].set_data(t, list(self._history["vel_err"]))
        self._lines["quat_err"].set_data(t, list(self._history["quat_err"]))
        self._lines["vel_kin_z"].set_data(t, list(self._history["vel_kin_z"]))
        left_state = "contact" if left_prob > 0.5 else "air"
        right_state = "contact" if right_prob > 0.5 else "air"
        self._contact_text.set_text(
            f"Left foot: {left_state} ({left_prob:.2f})\nRight foot: {right_state} ({right_prob:.2f})"
        )

        env_id = int(panel_data.get("env_id", 0))
        platform_quat_err = float(panel_data.get("platform_quat_error_rad", 0.0))
        platform_ang_vel_err = float(panel_data.get("platform_ang_vel_error", 0.0))
        platform_lin_acc_err = float(panel_data.get("platform_lin_acc_error", 0.0))
        self._axes[0].set_title(
            f"EKF vs Truth | env={env_id} | platform quat err={platform_quat_err:.3f} rad | "
            f"ang vel err={platform_ang_vel_err:.3f} | lin acc err={platform_lin_acc_err:.3f}"
        )

        for ax in self._axes:
            ax.relim()
            ax.autoscale_view()
            if t:
                ax.set_xlim(t[0], t[-1] if t[-1] > t[0] else t[0] + 1)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        if self.enabled and self._fig is not None and plt.fignum_exists(self._fig.number):
            plt.close(self._fig)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    # Optional: fix platform motion difficulty during play instead of using curriculum progression.
    if (
        args_cli.platform_motion_mode is not None
        or args_cli.platform_motion_level is not None
        or args_cli.platform_amp_scale is not None
    ):
        if getattr(env_cfg, "curriculum", None) is not None:
            if hasattr(env_cfg.curriculum, "platform_motion_levels"):
                env_cfg.curriculum.platform_motion_levels = None
            if hasattr(env_cfg.curriculum, "platform_motion_amplitude"):
                env_cfg.curriculum.platform_motion_amplitude = None

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Apply fixed platform settings for play, if provided.
    if args_cli.platform_motion_mode is not None:
        env.unwrapped.platform_motion_mode = str(args_cli.platform_motion_mode)
    if args_cli.platform_motion_level is not None:
        motion_mode = str(getattr(env.unwrapped, "platform_motion_mode", "xyz"))
        max_level = get_platform_motion_mode_max_level(motion_mode)
        if int(args_cli.platform_motion_level) > max_level:
            raise ValueError(
                f"platform_motion_level={args_cli.platform_motion_level} exceeds max level {max_level} "
                f"for motion mode '{motion_mode}'."
            )
        env.unwrapped.platform_motion_level = int(args_cli.platform_motion_level)
    if args_cli.platform_amp_scale is not None:
        env.unwrapped.platform_motion_amp_scale = float(min(max(args_cli.platform_amp_scale, 0.05), 1.0))

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    ekf_plotter = None
    if args_cli.ekf_panel and getattr(env_cfg, "ekf_debug_vis", False):
        if plt is None:
            print("[WARN] matplotlib is not available. EKF realtime panel is disabled.")
        else:
            ekf_plotter = EkfRealtimePlotter()

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, extras = env.step(actions)

        if ekf_plotter is not None:
            ekf_plotter.update(extras, timestep)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if ekf_plotter is not None:
        ekf_plotter.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
