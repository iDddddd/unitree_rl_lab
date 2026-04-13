from __future__ import annotations

import statistics
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.runners import OnPolicyRunner


def _build_activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


class _CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class ActorCriticROA(nn.Module):
    """Actor-critic with a jointly-trained ROA estimator."""

    is_recurrent = False
    supports_flat_export = False
    derived_estimator_feature_dim = 3

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        estimator_history_length: int = 20,
        estimator_input_dim: int = 44,
        estimator_frame_hidden_dim: int = 64,
        estimator_conv_channels: int = 128,
        estimator_hidden_dim: int = 128,
        estimator_output_dim: int = 11,
        policy_obs_set: str = "policy",
        critic_obs_set: str = "critic",
        estimator_history_set: str = "estimator_history",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticROA.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        actor_hidden_dims = actor_hidden_dims or [512, 256, 128]
        critic_hidden_dims = critic_hidden_dims or [512, 256, 128]

        self.obs_groups = obs_groups
        self.policy_obs_set = policy_obs_set
        self.critic_obs_set = critic_obs_set
        self.estimator_history_set = estimator_history_set
        self.estimator_history_length = estimator_history_length
        self.estimator_input_dim = estimator_input_dim
        self.estimator_output_dim = estimator_output_dim
        self.noise_std_type = noise_std_type

        num_policy_obs = self._obs_dim_from_set(obs, self.policy_obs_set)
        num_critic_obs = self._obs_dim_from_set(obs, self.critic_obs_set)
        num_history_obs = self._obs_dim_from_set(obs, self.estimator_history_set)
        expected_history_dim = estimator_history_length * estimator_input_dim
        if num_history_obs != expected_history_dim:
            raise ValueError(
                f"Estimator history dim mismatch: got {num_history_obs}, expected {expected_history_dim} "
                f"({estimator_history_length} x {estimator_input_dim})."
            )

        estimator_activation = _build_activation(activation)
        self.frame_encoder = nn.Sequential(
            nn.Linear(estimator_input_dim, estimator_frame_hidden_dim),
            _build_activation(activation),
        )
        self.temporal_conv1 = _CausalConv1d(
            estimator_frame_hidden_dim,
            estimator_conv_channels,
            kernel_size=5,
            dilation=1,
        )
        self.temporal_conv2 = _CausalConv1d(
            estimator_conv_channels,
            estimator_conv_channels,
            kernel_size=3,
            dilation=2,
        )
        self.temporal_conv1_act = _build_activation(activation)
        self.temporal_conv2_act = _build_activation(activation)
        self.estimator_backbone = nn.Sequential(
            nn.Linear(estimator_conv_channels, estimator_hidden_dim),
            estimator_activation,
        )
        self.contact_head = nn.Linear(estimator_hidden_dim, 2)
        self.base_velocity_head = nn.Linear(estimator_hidden_dim, 3)
        self.platform_linear_velocity_head = nn.Linear(estimator_hidden_dim, 3)
        self.platform_angular_velocity_head = nn.Linear(estimator_hidden_dim, 3)

        actor_input_dim = num_policy_obs + estimator_output_dim + self.derived_estimator_feature_dim
        self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(actor_input_dim)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'."
            )

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"ROA estimator history dim: {num_history_obs}")
        print(f"ROA actor MLP: {self.actor}")
        print(f"ROA critic MLP: {self.critic}")

    def _obs_dim_from_set(self, obs, set_name: str) -> int:
        total_dim = 0
        for obs_group in self.obs_groups[set_name]:
            assert len(obs[obs_group].shape) == 2, "ActorCriticROA only supports 1D observation groups."
            total_dim += obs[obs_group].shape[-1]
        return total_dim

    def _concat_obs_set(self, obs, set_name: str) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups[set_name]], dim=-1)

    def _forward_estimator_from_history(self, history_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = history_obs.shape[0]
        history_obs = history_obs.view(batch_size, self.estimator_history_length, self.estimator_input_dim)
        features = self.frame_encoder(history_obs)
        features = features.transpose(1, 2)
        features = self.temporal_conv1_act(self.temporal_conv1(features))
        features = self.temporal_conv2_act(self.temporal_conv2(features))
        features = self.estimator_backbone(features[:, :, -1])

        contact_logits = self.contact_head(features)
        base_lin_vel = self.base_velocity_head(features)
        platform_lin_vel = self.platform_linear_velocity_head(features)
        platform_ang_vel = self.platform_angular_velocity_head(features)

        estimator_features = torch.cat(
            [
                torch.sigmoid(contact_logits),
                base_lin_vel,
                platform_lin_vel,
                platform_ang_vel,
            ],
            dim=-1,
        )
        return {
            "contact_logits": contact_logits,
            "base_lin_vel": base_lin_vel,
            "platform_lin_vel": platform_lin_vel,
            "platform_ang_vel": platform_ang_vel,
            "features": estimator_features,
        }

    def forward_estimator(self, obs) -> dict[str, torch.Tensor]:
        history_obs = self._concat_obs_set(obs, self.estimator_history_set)
        return self._forward_estimator_from_history(history_obs)

    def _get_current_history_frame(self, obs) -> torch.Tensor:
        history_obs = self._concat_obs_set(obs, self.estimator_history_set)
        batch_size = history_obs.shape[0]
        history_obs = history_obs.view(batch_size, self.estimator_history_length, self.estimator_input_dim)
        return history_obs[:, -1, :]

    def _compute_relative_motion_features(
        self, obs, estimator_outputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        current_history_frame = self._get_current_history_frame(obs)
        current_base_ang_vel_z = current_history_frame[:, 5:6]
        planar_relative_velocity = estimator_outputs["platform_lin_vel"][:, :2] - estimator_outputs["base_lin_vel"][:, :2]
        relative_yaw_rate = estimator_outputs["platform_ang_vel"][:, 2:3] - current_base_ang_vel_z
        return torch.cat([planar_relative_velocity, relative_yaw_rate], dim=-1)

    def get_actor_obs(self, obs):
        policy_obs = self._concat_obs_set(obs, self.policy_obs_set)
        estimator_outputs = self.forward_estimator(obs)
        relative_motion_features = self._compute_relative_motion_features(obs, estimator_outputs)
        return torch.cat([policy_obs, estimator_outputs["features"], relative_motion_features], dim=-1)

    def get_critic_obs(self, obs):
        return self._concat_obs_set(obs, self.critic_obs_set)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'."
            )
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(actor_obs)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


class ROAPPO(PPO):
    """PPO with estimator supervision for single-stage ROA training."""

    def __init__(
        self,
        policy,
        contact_loss_coef: float = 1.0,
        base_velocity_loss_coef: float = 1.0,
        platform_linear_velocity_loss_coef: float = 1.5,
        platform_angular_velocity_loss_coef: float = 1.5,
        no_contact_loss_scale: float = 0.3,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise NotImplementedError("ROAPPO currently supports PPO-only training without RND or symmetry losses.")
        self.contact_loss_coef = contact_loss_coef
        self.base_velocity_loss_coef = base_velocity_loss_coef
        self.platform_linear_velocity_loss_coef = platform_linear_velocity_loss_coef
        self.platform_angular_velocity_loss_coef = platform_angular_velocity_loss_coef
        self.no_contact_loss_scale = no_contact_loss_scale

    def _compute_estimator_losses(self, obs_batch):
        target = obs_batch["estimator_target"]
        predictions = self.policy.forward_estimator(obs_batch)
        predicted_relative_motion = self.policy._compute_relative_motion_features(obs_batch, predictions)

        target_contact = target[:, 0:2]
        target_base_lin_vel = target[:, 2:5]
        target_platform_lin_vel = target[:, 5:8]
        target_platform_ang_vel = target[:, 8:11]
        current_history_frame = self.policy._get_current_history_frame(obs_batch)
        target_relative_motion = torch.cat(
            [
                target_platform_lin_vel[:, :2] - target_base_lin_vel[:, :2],
                target_platform_ang_vel[:, 2:3] - current_history_frame[:, 5:6],
            ],
            dim=-1,
        )

        contact_loss = F.binary_cross_entropy_with_logits(predictions["contact_logits"], target_contact)
        base_velocity_loss = F.smooth_l1_loss(predictions["base_lin_vel"], target_base_lin_vel)

        contact_mask = (target_contact.sum(dim=-1, keepdim=True) > 0.5).float()
        platform_weight = self.no_contact_loss_scale + (1.0 - self.no_contact_loss_scale) * contact_mask
        platform_linear_velocity_loss = (
            F.smooth_l1_loss(
                predictions["platform_lin_vel"],
                target_platform_lin_vel,
                reduction="none",
            ).mean(dim=-1, keepdim=True)
            * platform_weight
        ).mean()
        platform_angular_velocity_loss = (
            F.smooth_l1_loss(
                predictions["platform_ang_vel"],
                target_platform_ang_vel,
                reduction="none",
            ).mean(dim=-1, keepdim=True)
            * platform_weight
        ).mean()

        total_estimator_loss = (
            self.contact_loss_coef * contact_loss
            + self.base_velocity_loss_coef * base_velocity_loss
            + self.platform_linear_velocity_loss_coef * platform_linear_velocity_loss
            + self.platform_angular_velocity_loss_coef * platform_angular_velocity_loss
        )
        with torch.no_grad():
            contact_accuracy = (
                (torch.sigmoid(predictions["contact_logits"]) > 0.5) == (target_contact > 0.5)
            ).float().mean()
            base_velocity_mae = torch.mean(torch.abs(predictions["base_lin_vel"] - target_base_lin_vel))
            platform_linear_velocity_mae = torch.mean(
                torch.abs(predictions["platform_lin_vel"] - target_platform_lin_vel)
            )
            platform_angular_velocity_mae = torch.mean(
                torch.abs(predictions["platform_ang_vel"] - target_platform_ang_vel)
            )
            relative_lin_vel_x_mean = predicted_relative_motion[:, 0].mean()
            relative_lin_vel_y_mean = predicted_relative_motion[:, 1].mean()
            relative_yaw_rate_mean = predicted_relative_motion[:, 2].mean()
            relative_lin_vel_x_mae = torch.mean(
                torch.abs(predicted_relative_motion[:, 0] - target_relative_motion[:, 0])
            )
            relative_lin_vel_y_mae = torch.mean(
                torch.abs(predicted_relative_motion[:, 1] - target_relative_motion[:, 1])
            )
            relative_yaw_rate_mae = torch.mean(
                torch.abs(predicted_relative_motion[:, 2] - target_relative_motion[:, 2])
            )
        return total_estimator_loss, {
            "total": total_estimator_loss.item(),
            "contact": contact_loss.item(),
            "base_velocity": base_velocity_loss.item(),
            "platform_linear_velocity": platform_linear_velocity_loss.item(),
            "platform_angular_velocity": platform_angular_velocity_loss.item(),
            "contact_accuracy": contact_accuracy.item(),
            "base_velocity_mae": base_velocity_mae.item(),
            "platform_linear_velocity_mae": platform_linear_velocity_mae.item(),
            "platform_angular_velocity_mae": platform_angular_velocity_mae.item(),
            "actor_input_relative_lin_vel_x": relative_lin_vel_x_mean.item(),
            "actor_input_relative_lin_vel_y": relative_lin_vel_y_mean.item(),
            "actor_input_relative_yaw_rate": relative_yaw_rate_mean.item(),
            "actor_input_relative_lin_vel_x_mae": relative_lin_vel_x_mae.item(),
            "actor_input_relative_lin_vel_y_mae": relative_lin_vel_y_mae.item(),
            "actor_input_relative_yaw_rate_mae": relative_yaw_rate_mae.item(),
        }

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimator_total_loss = 0.0
        mean_estimator_contact_loss = 0.0
        mean_estimator_base_velocity_loss = 0.0
        mean_estimator_platform_linear_velocity_loss = 0.0
        mean_estimator_platform_angular_velocity_loss = 0.0
        mean_estimator_contact_accuracy = 0.0
        mean_estimator_base_velocity_mae = 0.0
        mean_estimator_platform_linear_velocity_mae = 0.0
        mean_estimator_platform_angular_velocity_mae = 0.0
        mean_actor_input_relative_lin_vel_x = 0.0
        mean_actor_input_relative_lin_vel_y = 0.0
        mean_actor_input_relative_yaw_rate = 0.0
        mean_actor_input_relative_lin_vel_x_mae = 0.0
        mean_actor_input_relative_lin_vel_y_mae = 0.0
        mean_actor_input_relative_yaw_rate_mae = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            estimator_loss, estimator_metrics = self._compute_estimator_losses(obs_batch)
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + estimator_loss
            )

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_estimator_total_loss += estimator_metrics["total"]
            mean_estimator_contact_loss += estimator_metrics["contact"]
            mean_estimator_base_velocity_loss += estimator_metrics["base_velocity"]
            mean_estimator_platform_linear_velocity_loss += estimator_metrics["platform_linear_velocity"]
            mean_estimator_platform_angular_velocity_loss += estimator_metrics["platform_angular_velocity"]
            mean_estimator_contact_accuracy += estimator_metrics["contact_accuracy"]
            mean_estimator_base_velocity_mae += estimator_metrics["base_velocity_mae"]
            mean_estimator_platform_linear_velocity_mae += estimator_metrics["platform_linear_velocity_mae"]
            mean_estimator_platform_angular_velocity_mae += estimator_metrics["platform_angular_velocity_mae"]
            mean_actor_input_relative_lin_vel_x += estimator_metrics["actor_input_relative_lin_vel_x"]
            mean_actor_input_relative_lin_vel_y += estimator_metrics["actor_input_relative_lin_vel_y"]
            mean_actor_input_relative_yaw_rate += estimator_metrics["actor_input_relative_yaw_rate"]
            mean_actor_input_relative_lin_vel_x_mae += estimator_metrics["actor_input_relative_lin_vel_x_mae"]
            mean_actor_input_relative_lin_vel_y_mae += estimator_metrics["actor_input_relative_lin_vel_y_mae"]
            mean_actor_input_relative_yaw_rate_mae += estimator_metrics["actor_input_relative_yaw_rate_mae"]

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimator_total_loss /= num_updates
        mean_estimator_contact_loss /= num_updates
        mean_estimator_base_velocity_loss /= num_updates
        mean_estimator_platform_linear_velocity_loss /= num_updates
        mean_estimator_platform_angular_velocity_loss /= num_updates
        mean_estimator_contact_accuracy /= num_updates
        mean_estimator_base_velocity_mae /= num_updates
        mean_estimator_platform_linear_velocity_mae /= num_updates
        mean_estimator_platform_angular_velocity_mae /= num_updates
        mean_actor_input_relative_lin_vel_x /= num_updates
        mean_actor_input_relative_lin_vel_y /= num_updates
        mean_actor_input_relative_yaw_rate /= num_updates
        mean_actor_input_relative_lin_vel_x_mae /= num_updates
        mean_actor_input_relative_lin_vel_y_mae /= num_updates
        mean_actor_input_relative_yaw_rate_mae /= num_updates

        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator_total": mean_estimator_total_loss,
            "estimator_contact_loss": mean_estimator_contact_loss,
            "estimator_base_velocity_loss": mean_estimator_base_velocity_loss,
            "estimator_platform_linear_velocity_loss": mean_estimator_platform_linear_velocity_loss,
            "estimator_platform_angular_velocity_loss": mean_estimator_platform_angular_velocity_loss,
            "estimator_contact_accuracy": mean_estimator_contact_accuracy,
            "estimator_base_velocity_mae": mean_estimator_base_velocity_mae,
            "estimator_platform_linear_velocity_mae": mean_estimator_platform_linear_velocity_mae,
            "estimator_platform_angular_velocity_mae": mean_estimator_platform_angular_velocity_mae,
            "actor_input_relative_lin_vel_x": mean_actor_input_relative_lin_vel_x,
            "actor_input_relative_lin_vel_y": mean_actor_input_relative_lin_vel_y,
            "actor_input_relative_yaw_rate": mean_actor_input_relative_yaw_rate,
            "actor_input_relative_lin_vel_x_mae": mean_actor_input_relative_lin_vel_x_mae,
            "actor_input_relative_lin_vel_y_mae": mean_actor_input_relative_lin_vel_y_mae,
            "actor_input_relative_yaw_rate_mae": mean_actor_input_relative_yaw_rate_mae,
        }


class ROAOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for single-stage ROA training."""

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        estimator_lines = []
        actor_input_lines = []
        for key, value in locs["loss_dict"].items():
            if key.startswith("estimator_"):
                metric_name = key[len("estimator_") :]
                self.writer.add_scalar(f"Estimator/{metric_name}", value, locs["it"])
                estimator_lines.append(f"""{f'{metric_name}:':>{pad}} {value:.4f}\n""")
            elif key.startswith("actor_input_"):
                metric_name = key[len("actor_input_") :]
                self.writer.add_scalar(f"ActorInput/{metric_name}", value, locs["it"])
                actor_input_lines.append(f"""{f'{metric_name}:':>{pad}} {value:.4f}\n""")
            else:
                self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])

        if locs["rewbuffer"]:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            log_string = (
                f"""{'#' * width}\n"""
                f"""{''.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['loss_dict']['value_function']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['loss_dict']['surrogate']:.4f}\n"""
                f"""{'Entropy:':>{pad}} {locs['loss_dict']['entropy']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{''.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                if not key.startswith("estimator_"):
                    log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        if estimator_lines:
            log_string += f"""{'-' * width}\n"""
            log_string += "Estimator\n"
            log_string += "".join(estimator_lines)
        if actor_input_lines:
            log_string += f"""{'-' * width}\n"""
            log_string += "ActorInput\n"
            log_string += "".join(actor_input_lines)
        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                '%H:%M:%S',
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def _construct_algorithm(self, obs):
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        policy_cfg = dict(self.policy_cfg)
        alg_cfg = dict(self.alg_cfg)

        policy_class_name = policy_cfg.pop("class_name")
        policy_classes = {
            "ActorCriticROA": ActorCriticROA,
            "ActorCritic": ActorCritic,
        }
        if policy_class_name not in policy_classes:
            raise ValueError(f"Unsupported ROA policy class: {policy_class_name}")
        actor_critic = policy_classes[policy_class_name](
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
        ).to(self.device)

        alg_class_name = alg_cfg.pop("class_name")
        alg_classes = {
            "ROAPPO": ROAPPO,
        }
        if alg_class_name not in alg_classes:
            raise ValueError(f"Unsupported ROA algorithm class: {alg_class_name}")
        alg = alg_classes[alg_class_name](
            actor_critic,
            device=self.device,
            **alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return alg
