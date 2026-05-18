# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete
import random

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import numpy as np
import os
import shutil
import torch
from copy import deepcopy
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


ENV_STATE_INFO_KEY = "unitree_rl_lab_env_state"
RUNNER_INFOS_KEY = "rsl_rl_runner_infos"
RUNNER_STATE_INFO_KEY = "unitree_rl_lab_runner_state"


def _to_cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return value


def _restore_tensor_attr(obj, attr_name: str, value, device: torch.device | str):
    if value is None:
        return
    value_t = torch.as_tensor(value, device=device)
    current = getattr(obj, attr_name, None)
    if isinstance(current, torch.Tensor) and current.shape == value_t.shape:
        current.copy_(value_t.to(dtype=current.dtype))
    else:
        setattr(obj, attr_name, value_t)


def _collect_named_tensor_attrs(obj, attr_names: list[str]) -> dict[str, torch.Tensor]:
    state = {}
    for attr_name in attr_names:
        value = getattr(obj, attr_name, None)
        if isinstance(value, torch.Tensor):
            state[attr_name] = _to_cpu_clone(value)
    return state


def _collect_command_state(env) -> dict:
    if not hasattr(env, "command_manager"):
        return {}

    command_state = {}
    for term_name in env.command_manager.active_terms:
        term = env.command_manager.get_term(term_name)
        term_state = {
            "tensors": _collect_named_tensor_attrs(term, ["time_left", "command_counter"]),
            "cfg": {},
        }
        try:
            command = getattr(term, "command", None)
        except Exception:
            command = None
        if isinstance(command, torch.Tensor):
            term_state["command"] = _to_cpu_clone(command)

        ranges = getattr(term.cfg, "ranges", None)
        if ranges is not None:
            term_state["cfg"]["ranges"] = deepcopy(ranges.to_dict() if hasattr(ranges, "to_dict") else ranges)
        limit_ranges = getattr(term.cfg, "limit_ranges", None)
        if limit_ranges is not None:
            term_state["cfg"]["limit_ranges"] = deepcopy(
                limit_ranges.to_dict() if hasattr(limit_ranges, "to_dict") else limit_ranges
            )
        command_state[term_name] = term_state
    return command_state


def _restore_command_state(env, command_state: dict):
    if not command_state or not hasattr(env, "command_manager"):
        return

    for term_name, term_state in command_state.items():
        if term_name not in env.command_manager.active_terms:
            continue
        term = env.command_manager.get_term(term_name)
        for attr_name, value in term_state.get("tensors", {}).items():
            _restore_tensor_attr(term, attr_name, value, env.device)

        command = term_state.get("command")
        if command is not None:
            command_t = torch.as_tensor(command, device=env.device)
            # CommandTerm.command is a read-only property, but most terms store it in a private tensor.
            for candidate in ("command", "_command", "vel_command_b", "_vel_command_b"):
                current = getattr(term, candidate, None)
                if isinstance(current, torch.Tensor) and current.shape == command_t.shape:
                    current.copy_(command_t.to(dtype=current.dtype))
                    break

        cfg_state = term_state.get("cfg", {})
        ranges_state = cfg_state.get("ranges")
        if ranges_state is not None and hasattr(term.cfg, "ranges"):
            for key, value in ranges_state.items():
                setattr(term.cfg.ranges, key, value)


def _collect_reward_state(env) -> dict:
    if not hasattr(env, "reward_manager"):
        return {}

    weights = {}
    for term_name in env.reward_manager.active_terms:
        weights[term_name] = float(env.reward_manager.get_term_cfg(term_name).weight)
    episode_sums = {
        term_name: _to_cpu_clone(value)
        for term_name, value in getattr(env.reward_manager, "_episode_sums", {}).items()
        if isinstance(value, torch.Tensor)
    }
    return {"weights": weights, "episode_sums": episode_sums}


def _restore_reward_state(env, reward_state: dict):
    if not reward_state or not hasattr(env, "reward_manager"):
        return

    for term_name, weight in reward_state.get("weights", {}).items():
        if term_name in env.reward_manager.active_terms:
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            term_cfg.weight = float(weight)
            env.reward_manager.set_term_cfg(term_name, term_cfg)

    for term_name, value in reward_state.get("episode_sums", {}).items():
        current = getattr(env.reward_manager, "_episode_sums", {}).get(term_name)
        if isinstance(current, torch.Tensor):
            value_t = torch.as_tensor(value, device=env.device, dtype=current.dtype)
            if current.shape == value_t.shape:
                current.copy_(value_t)


def _collect_env_resume_state(env) -> dict:
    """Collect environment-side training state that RSL-RL does not store in its checkpoint."""

    scalar_attrs = [
        "common_step_counter",
        "platform_motion_level",
        "platform_motion_amp_scale",
        "platform_motion_mode",
        "platform_level_start_episode",
        "platform_curriculum_score",
        "platform_curriculum_score_ema",
        "platform_reward_profile_id",
    ]
    tensor_attrs = [
        "episode_length_buf",
        "_platform_motion_phase",
        "_platform_base_root_state_w",
    ]

    state = {
        "version": 1,
        "scalars": {name: getattr(env, name) for name in scalar_attrs if hasattr(env, name)},
        "tensors": _collect_named_tensor_attrs(env, tensor_attrs),
        "commands": _collect_command_state(env),
        "rewards": _collect_reward_state(env),
    }
    return state


def _restore_env_resume_state(env, infos, fallback_common_step_counter: int | None = None):
    """Restore environment-side training state from checkpoint infos."""

    if isinstance(infos, dict) and ENV_STATE_INFO_KEY in infos:
        state = infos[ENV_STATE_INFO_KEY]
    else:
        state = None

    if not state:
        print("[INFO] No Unitree env resume state found in checkpoint; using freshly initialized env curriculum.")
        if fallback_common_step_counter is not None:
            env.common_step_counter = int(fallback_common_step_counter)
            print(f"[INFO] Estimated env common_step_counter from runner iteration: {env.common_step_counter}")
        return

    for attr_name, value in state.get("scalars", {}).items():
        setattr(env, attr_name, value)

    for attr_name, value in state.get("tensors", {}).items():
        _restore_tensor_attr(env, attr_name, value, env.device)

    _restore_command_state(env, state.get("commands", {}))
    _restore_reward_state(env, state.get("rewards", {}))

    level = getattr(env, "platform_motion_level", None)
    amp_scale = getattr(env, "platform_motion_amp_scale", None)
    step = getattr(env, "common_step_counter", None)
    print(f"[INFO] Restored Unitree env state: step={step}, platform_level={level}, amp_scale={amp_scale}")


def _merge_runner_infos_with_env_state(infos, env_state: dict) -> dict:
    if isinstance(infos, dict):
        merged_infos = dict(infos)
        previous_runner_infos = merged_infos.pop(RUNNER_INFOS_KEY, None)
        if previous_runner_infos is not None and RUNNER_INFOS_KEY not in merged_infos:
            merged_infos[RUNNER_INFOS_KEY] = previous_runner_infos
    else:
        merged_infos = {}
        if infos is not None:
            merged_infos[RUNNER_INFOS_KEY] = infos
    merged_infos[ENV_STATE_INFO_KEY] = env_state
    return merged_infos


def _collect_runner_resume_state(runner) -> dict:
    state = {
        "tot_timesteps": int(getattr(runner, "tot_timesteps", 0)),
        "tot_time": float(getattr(runner, "tot_time", 0.0)),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    if torch.cuda.is_available():
        state["rng"]["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_runner_resume_state(runner, infos):
    if not isinstance(infos, dict):
        return
    state = infos.get(RUNNER_STATE_INFO_KEY)
    if not state:
        return
    runner.tot_timesteps = int(state.get("tot_timesteps", getattr(runner, "tot_timesteps", 0)))
    runner.tot_time = float(state.get("tot_time", getattr(runner, "tot_time", 0.0)))
    rng_state = state.get("rng", {})
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def _patch_runner_env_state_checkpointing(runner):
    """Attach env-state checkpointing to this runner instance without modifying RSL-RL itself."""

    original_save = runner.save

    def save_with_env_state(path: str, infos=None):
        env_state = _collect_env_resume_state(runner.env.unwrapped)
        merged_infos = _merge_runner_infos_with_env_state(infos, env_state)
        merged_infos[RUNNER_STATE_INFO_KEY] = _collect_runner_resume_state(runner)
        return original_save(path, infos=merged_infos)

    runner.save = save_with_env_state


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    _patch_runner_env_state_checkpointing(runner)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        checkpoint_infos = runner.load(resume_path)
        _restore_runner_resume_state(runner, checkpoint_infos)
        fallback_step = int(max(runner.current_learning_iteration + 1, 0) * runner.num_steps_per_env)
        _restore_env_resume_state(env.unwrapped, checkpoint_infos, fallback_common_step_counter=fallback_step)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    randomize_episode_lengths = not (agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=randomize_episode_lengths)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
