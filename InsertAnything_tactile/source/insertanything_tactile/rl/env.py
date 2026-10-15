"""Six-DOF insertion with a frozen stereo angle estimator in observations/rewards."""
import math
import torch
from insertanything_tactile.env import InsertAnythingTactileEnv
from tactile_pose.rl_estimator import FrozenNormalEstimator
from tactile_pose.rl_math import (quat_apply, quat_inverse,
    world_delta_target, estimated_axis_w, local_z_w, axis_angle, gaussian_alignment, measured_in_hand)
from InsertAnything.tasks.direct.insertanything import insertanything_utils


class Tactile6DEnv(InsertAnythingTactileEnv):
    EXTRA_OBSERVATIONS = 5  # theta/15, estimated object axis in hole frame (3), valid

    def __init__(self, cfg, **kwargs):
        if tuple(cfg.tactile.sides) != ("left", "right"):
            raise ValueError("Tactile6D requires both calibrated sensors")
        a = cfg.tactile_alignment
        if not 0 <= a.initial_tilt_deg <= 15:
            raise ValueError("Initial in-hand tilt must be within the model's +/-15 degree range")
        if min(a.sigma_deg, a.insertion_gate_sigma_deg, a.max_rotation_step_rad) <= 0:
            raise ValueError("Reward widths and rotation limit must be positive")
        # Match the original robot's new initialization: upright wrist, with
        # the peg tilted about TCP +Y by the shared upstream reset routine.
        cfg.task.hand_init_orn[:2] = [math.pi, 0.0]
        cfg.task.hand_init_orn_range[:2] = [0.0, 0.0]
        cfg.initial_in_hand_tilt_deg = a.initial_tilt_deg
        cfg.ctrl.lock_roll_pitch = False
        self.initialization_metadata = {
            "mode": "upright_tcp_tilted_peg_v2", "tilt_half_range_deg": float(a.initial_tilt_deg),
            "axis_frame": "TCP", "axis": [0.0, 1.0, 0.0], "pivot": "TCP",
            "grasp_anchor": "object_mesh_center",
        }
        self._prediction_token = None
        self._reset_version = 0
        super().__init__(cfg, **kwargs)
        self.angle_estimator = FrozenNormalEstimator(a.checkpoint, self.tactile_calibration, self.device)
        self._finger_joint_ids = [self._robot.joint_names.index(f"panda_finger_joint{i}") for i in (1, 2)]
        # Use exactly the same mesh-centre convention as the v4 collection.
        from tactile_pose.tcp_sim_collection import mesh_points
        pts = mesh_points(self.sim.stage, "/World/envs/env_0/HeldAsset")
        self._object_center_o = torch.tensor((pts.min(0)+pts.max(0))/2, device=self.device, dtype=torch.float32)

    def _configure_gym_env_spaces(self):
        # Called once from DirectRLEnv after upstream computes per-task dimensions.
        self.cfg.observation_space += self.EXTRA_OBSERVATIONS
        self.cfg.state_space += self.EXTRA_OBSERVATIONS
        super()._configure_gym_env_spaces()

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        # With wrist tilt enabled the linear Jacobian must refer to the same
        # TCP used by the pose error, rather than the average finger origins.
        index = self.fingertip_body_idx - int(self._robot.is_fixed_base)
        self.fingertip_midpoint_jacobian = self._robot.root_physx_view.get_jacobians()[:, index, :6, :7]

    def get_handheld_asset_relative_pose(self):
        # Keep the calibrated gelpad grasp position. The nominal relative
        # orientation and the subsequent TCP-pivot tilt both come from upstream.
        _, returned_q = super().get_handheld_asset_relative_pose()
        self._reset_grasp_pos = self.fingertip_midpoint_pos.clone()
        self._reset_grasp_quat = self.fingertip_midpoint_quat.clone()
        return self._object_center_o.expand(self.num_envs, 3).clone(), returned_q

    def close_gripper_in_place(self):
        # Hold the upright wrist while the fingers physically grasp the tilted peg.
        self.generate_ctrl_signals(self._reset_grasp_pos, self._reset_grasp_quat, 0.0)

    def _reset_idx(self, env_ids):
        # Upstream physically steps the whole scene and rewrites all held assets
        # while grasping. This task therefore deliberately uses synchronous,
        # timeout-only episodes. Reject unsupported partial resets explicitly.
        if len(env_ids) != self.num_envs:
            raise ValueError("Tactile6D uses synchronous full-batch resets; reset all environments together")
        super()._reset_idx(env_ids)
        self._reset_version += 1
        self._prediction_token = None

    def _apply_action(self):
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values()
        target_pos = self.fingertip_midpoint_pos + self.actions[:, :3]*self.pos_threshold
        centre = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        target_pos = centre + (target_pos-centre).clamp(-bounds, bounds)
        target_quat = world_delta_target(self.fingertip_midpoint_quat,
            self.actions[:, 3:]*self.rot_threshold, self.cfg.tactile_alignment.max_rotation_step_rad)
        # All three world-frame rotation-vector components reach the controller.
        self.rl_control_target_quat = target_quat
        self.generate_ctrl_signals(target_pos, target_quat, 0.0)

    def _get_dones(self):
        self._compute_intermediate_values()
        timeout = self.episode_length_buf >= self.max_episode_length
        return torch.zeros_like(timeout), timeout

    @torch.no_grad()
    def _ensure_prediction(self):
        token = (self.common_step_counter, self._reset_version, float(self._robot._data._sim_timestamp))
        if self._prediction_token == token:
            return
        self._compute_intermediate_values()
        # Read synchronized post-physics images, once for reward + observations.
        tactile = self.refresh_tactile(render=True)
        opening = self._robot.data.joint_pos[:, self._finger_joint_ids].sum(-1)
        a = self.cfg.tactile_alignment
        self.predicted_angle_deg, self.tactile_valid = self.angle_estimator(
            tactile, opening, a.min_contact_pixels, a.min_contact_depth_mm)
        self.estimated_object_axis_w = estimated_axis_w(self.fingertip_midpoint_quat,
                                                       torch.deg2rad(self.predicted_angle_deg))
        self.hole_axis_w = local_z_w(self.fixed_quat)
        self.estimated_axis_error_rad = axis_angle(self.estimated_object_axis_w, self.hole_axis_w)
        axis_h = quat_apply(quat_inverse(self.fixed_quat), self.estimated_object_axis_w)
        valid = self.tactile_valid.float().unsqueeze(-1)
        self.tactile_policy_features = torch.cat((self.predicted_angle_deg[:, None]/15.,
                                                  axis_h*valid, valid), dim=-1)
        self._prediction_token = token

    def _get_observations(self):
        self._ensure_prediction()
        obs = super()._get_observations()
        return {key: torch.cat((value, self.tactile_policy_features), dim=-1) for key, value in obs.items()}

    def _get_curr_successes(self, success_threshold):
        self._ensure_prediction()
        original = super()._get_curr_successes(success_threshold)
        # Original test is one-sided in Z; reject objects passing far below the hole.
        held_p, _ = insertanything_utils.get_held_base_pose(self.held_pos, self.held_quat, self.num_envs, self.device)
        target_p, _ = insertanything_utils.get_target_held_base_pose(self.fixed_pos, self.fixed_quat,
                                                                    self.cfg_task, self.num_envs, self.device)
        bounded_depth = (held_p[:, 2]-target_p[:, 2]) >= -0.002
        return (original & bounded_depth & self.tactile_valid &
                (self.estimated_axis_error_rad < math.radians(self.cfg.tactile_alignment.success_angle_deg)))

    def _get_factory_rew_dict(self):
        self._ensure_prediction()
        terms, scales, success = super()._get_factory_rew_dict()
        a = self.cfg.tactile_alignment
        terms["tactile_axis_align"] = gaussian_alignment(self.estimated_axis_error_rad, self.tactile_valid, a.sigma_deg)
        scales["tactile_axis_align"] = a.weight
        terms["z_insert"] *= gaussian_alignment(self.estimated_axis_error_rad, self.tactile_valid,
                                                 a.insertion_gate_sigma_deg)
        terms["tactile_contact_loss"] = (~self.tactile_valid).float()
        scales["tactile_contact_loss"] = -a.contact_loss_penalty
        return terms, scales, success

    def _get_rewards(self):
        reward = super()._get_rewards()
        self.last_reward_diagnostics = {k: v.clone() for k, v in self.angle_diagnostics().items()}
        for key, value in self.last_reward_diagnostics.items():
            self.extras["tactile/"+key] = value.float().mean()
        valid = self.tactile_valid.float()
        self.extras["tactile/gt_prediction_mae_valid_deg"] = (
            self.last_reward_diagnostics["gt_prediction_abs_error_deg"]*valid).sum()/valid.sum().clamp_min(1)
        return reward

    def angle_diagnostics(self):
        """Truth is for logging only, separated from the estimated reward path."""
        self._ensure_prediction()
        truth, off_axis = measured_in_hand(self.fingertip_midpoint_quat, self.held_quat)
        true_alpha = axis_angle(local_z_w(self.held_quat), self.hole_axis_w)
        true_insert = super()._get_curr_successes(self.cfg_task.success_threshold)
        held_p, _ = insertanything_utils.get_held_base_pose(self.held_pos, self.held_quat, self.num_envs, self.device)
        target_p, _ = insertanything_utils.get_target_held_base_pose(self.fixed_pos, self.fixed_quat,
                                                                    self.cfg_task, self.num_envs, self.device)
        true_insert &= (held_p[:, 2]-target_p[:, 2] >= -.002)
        true_insert &= true_alpha < math.radians(self.cfg.tactile_alignment.success_angle_deg)
        true_insert &= self.tactile_valid
        return {"valid_fraction": self.tactile_valid,
                "predicted_angle_deg": self.predicted_angle_deg,
                "estimated_axis_error_deg": torch.rad2deg(self.estimated_axis_error_rad),
                "gt_in_hand_angle_deg": torch.rad2deg(truth),
                "gt_prediction_abs_error_deg": (self.predicted_angle_deg-torch.rad2deg(truth)).abs(),
                "gt_off_axis_rotation_deg": torch.rad2deg(off_axis),
                "gt_axis_error_deg": torch.rad2deg(true_alpha),
                "gt_axis_within_3deg": true_alpha < math.radians(3.),
                "estimated_insert_success": self._get_curr_successes(self.cfg_task.success_threshold),
                "gt_insert_success": true_insert,
                "prediction_near_limit": self.predicted_angle_deg.abs() > 14.5}
