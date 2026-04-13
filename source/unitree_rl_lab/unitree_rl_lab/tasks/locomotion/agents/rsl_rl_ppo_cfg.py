# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RoaActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "ActorCriticROA"
    actor_obs_normalization = False
    critic_obs_normalization = False
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    estimator_history_length = 20
    estimator_input_dim = 44
    estimator_frame_hidden_dim = 64
    estimator_conv_channels = 128
    estimator_hidden_dim = 128
    estimator_output_dim = 11


@configclass
class RoaPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "ROAPPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
    contact_loss_coef = 1.0
    base_velocity_loss_coef = 1.0
    platform_linear_velocity_loss_coef = 1.5
    platform_angular_velocity_loss_coef = 1.5
    no_contact_loss_scale = 0.3


@configclass
class RoaPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "ROAOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "estimator_history": ["estimator_history"],
        "estimator_target": ["estimator_target"],
    }
    policy = RoaActorCriticCfg()
    algorithm = RoaPpoAlgorithmCfg()
