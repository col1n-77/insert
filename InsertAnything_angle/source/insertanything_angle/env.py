"""No-tactile angle observations, axis reward, and optional depth-axis gating.

The optional gate changes only the original dense depth reward. Assets,
control, force penalty, stage bonuses, success and termination remain inherited.
"""

from __future__ import annotations

import math

import torch

from InsertAnything.tasks.direct.insertanything import insertanything_utils
from InsertAnything.tasks.direct.insertanything.insertanything_env import InsertAnythingEnv

from .env_cfg import InsertAnythingAngleEnvCfg
from .math import AngleNoiseCache, angle_geometry, gaussian_axis_reward, axis_depth_gate


class InsertAnythingAngleEnv(InsertAnythingEnv):
    cfg: InsertAnythingAngleEnvCfg

    def __init__(self, cfg: InsertAnythingAngleEnvCfg, render_mode: str | None = None, **kwargs):
        for name, value in (
            ("angle_obs.noise_std_deg", cfg.angle_obs.noise_std_deg),
            ("axis_alignment.reward_weight", cfg.axis_alignment.reward_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("angle_obs.scale_deg", cfg.angle_obs.scale_deg),
            ("axis_alignment.sigma_deg", cfg.axis_alignment.sigma_deg),
            ("axis_alignment.strict_axis_deg", cfg.axis_alignment.strict_axis_deg),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(cfg.axis_alignment.strict_min_depth_mm):
            raise ValueError("axis_alignment.strict_min_depth_mm must be finite")
        if not isinstance(cfg.axis_alignment.zero_at_alignment, bool):
            raise ValueError("axis_alignment.zero_at_alignment must be a boolean")
        if not isinstance(cfg.depth_axis_gate.enabled, bool):
            raise ValueError("depth_axis_gate.enabled must be a boolean")
        gate = cfg.depth_axis_gate
        axis_depth_gate(torch.zeros(1), gate.tolerance_deg, gate.sigma_deg, gate.min_gate)
        if gate.enabled and not cfg.task.use_decoupled_reward:
            raise ValueError("Depth-axis gating requires the task's decoupled z_insert reward")
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._angle_noise = AngleNoiseCache(
            self.num_envs, self.device, cfg.angle_obs.noise_std_deg,
            int(cfg.seed or 0) + int(cfg.angle_obs.noise_seed_offset),
        )
        self.angle_diagnostics: dict[str, torch.Tensor] = {}

    def _configure_gym_env_spaces(self):
        # The original constructor computes task-specific dimensions before
        # DirectRLEnv calls this method. Extend them before the spaces/buffers
        # are allocated, rather than trusting static config dimension values.
        self.base_policy_dim = int(self.cfg.observation_space)
        self.base_critic_dim = int(self.cfg.state_space)
        self.cfg.observation_space = self.base_policy_dim + 1
        self.cfg.state_space = self.base_critic_dim + int(self.cfg.angle_obs.critic_true_angle)
        super()._configure_gym_env_spaces()

    def seed(self, seed: int = -1) -> int:
        actual_seed = super().seed(seed)
        if hasattr(self, "_angle_noise"):
            self._angle_noise.reseed(actual_seed + int(self.cfg.angle_obs.noise_seed_offset))
        return actual_seed

    def _pre_physics_step(self, action):
        self._angle_noise.invalidate()
        super()._pre_physics_step(action)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_angle_noise"):
            self._angle_noise.invalidate(env_ids)

    def _angle_signals(self) -> dict[str, torch.Tensor]:
        signals = angle_geometry(self.held_quat, self.fingertip_midpoint_quat, self.fixed_quat)
        noise = self._angle_noise.sample()
        signals["measurement_noise_rad"] = noise.clone()
        signals["measured_in_hand_angle_rad"] = signals["true_in_hand_angle_rad"] + noise
        signals["noise_sample_count"] = self._angle_noise.sample_count.clone()
        signals["alignment_reward"] = gaussian_axis_reward(
            signals["axis_error_rad"], self.cfg.axis_alignment.reward_weight,
            self.cfg.axis_alignment.sigma_deg,
            zero_at_alignment=self.cfg.axis_alignment.zero_at_alignment,
        )
        for key in ("true_in_hand_angle", "measured_in_hand_angle", "measurement_noise", "axis_error", "off_axis_residual"):
            signals[f"{key}_deg"] = torch.rad2deg(signals[f"{key}_rad"])
        return signals

    def get_angle_diagnostics(self) -> dict[str, torch.Tensor]:
        """Current geometry and success diagnostics; no extra noise samples."""
        signals = self._angle_signals()
        held_base, _ = insertanything_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.num_envs, self.device,
        )
        target_base, _ = insertanything_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task, self.num_envs, self.device,
        )
        native = self._get_curr_successes(self.cfg_task.success_threshold)
        depth_mm = (held_base[:, 2] - target_base[:, 2]) * 1000.0
        signals["native_success"] = native
        signals["depth_error_mm"] = depth_mm
        signals["xy_error_mm"] = torch.linalg.vector_norm(held_base[:, :2] - target_base[:, :2], dim=-1) * 1000.0
        signals["strict_success"] = (
            native
            & (signals["axis_error_deg"] < self.cfg.axis_alignment.strict_axis_deg)
            & (depth_mm >= self.cfg.axis_alignment.strict_min_depth_mm)
        )
        self.angle_diagnostics = signals
        return signals

    def _get_observations(self):
        observations = super()._get_observations()
        signals = self._angle_signals()
        scale_rad = math.radians(self.cfg.angle_obs.scale_deg)
        # Do not clamp: the existing RL wrapper handles its ordinary clipping.
        observations["policy"] = torch.cat(
            (observations["policy"], (signals["measured_in_hand_angle_rad"] / scale_rad).unsqueeze(-1)), dim=-1,
        )
        if self.cfg.angle_obs.critic_true_angle:
            observations["critic"] = torch.cat(
                (observations["critic"], (signals["true_in_hand_angle_rad"] / scale_rad).unsqueeze(-1)), dim=-1,
            )
        return observations

    def _get_rewards(self):
        original = super()._get_rewards()
        signals = self.get_angle_diagnostics()
        depth_original = self._depth_reward_before_axis_gate
        gate_cfg = self.cfg.depth_axis_gate
        gate = (axis_depth_gate(signals["axis_error_rad"], gate_cfg.tolerance_deg,
                                gate_cfg.sigma_deg, gate_cfg.min_gate)
                if gate_cfg.enabled else torch.ones_like(original))
        adjustment = (gate - 1.0) * depth_original
        total = original + signals["alignment_reward"]
        if gate_cfg.enabled:
            total = total + adjustment
        signals["original_reward"] = original.detach().clone()
        signals["depth_axis_gate"] = gate.detach().clone()
        signals["depth_reward_before_axis_gate"] = depth_original.detach().clone()
        signals["depth_reward_after_axis_gate"] = (gate * depth_original).detach().clone()
        signals["depth_gate_reward_adjustment"] = adjustment.detach().clone()
        signals["total_reward"] = total.detach().clone()
        self.extras["logs_rew_axis_alignment"] = signals["alignment_reward"].mean()
        self.extras["logs_rew_depth_gate_adjustment"] = adjustment.mean()
        self.extras["logs_rew_depth_after_axis_gate"] = signals["depth_reward_after_axis_gate"].mean()
        self.extras["logs_angle_depth_axis_gate"] = gate.mean()
        self.extras["logs_angle_axis_error_deg"] = signals["axis_error_deg"].mean()
        self.extras["logs_angle_true_abs_deg"] = signals["true_in_hand_angle_deg"].abs().mean()
        self.extras["logs_angle_off_axis_deg"] = signals["off_axis_residual_deg"].mean()
        self.extras["logs_angle_strict_success"] = signals["strict_success"].float().mean()
        return total

    def _get_factory_rew_dict(self):
        rewards, scales, successes = super()._get_factory_rew_dict()
        # Capture the weighted depth term from the SAME evaluation. Do not call
        # the base reward twice: it updates prev_actions and episode logs.
        depth = rewards.get("z_insert")
        if depth is None:
            if self.cfg.depth_axis_gate.enabled:
                raise RuntimeError("Enabled depth-axis gate has no z_insert term")
            self._depth_reward_before_axis_gate = torch.zeros_like(successes, dtype=torch.float32)
        else:
            self._depth_reward_before_axis_gate = depth * scales["z_insert"]
        return rewards, scales, successes
