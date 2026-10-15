"""Finite physical contract check for the separate no-tactile angle task.

Runs zero-action steps, not RL training. Use the project's scripts/python.sh
after selecting the existing Isaac runtime, and provide a new JSON output path.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="InsertAnythingAngle-LHole-III-Direct-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--seed", type=int, default=991)
parser.add_argument("--depth_axis_gate", action="store_true")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output is None or args.num_envs < 1 or args.steps < 1:
    parser.error("--output and positive --num_envs/--steps are required")
if not args.task.startswith("InsertAnythingAngle-"):
    parser.error("Use an independent InsertAnythingAngle task")
args.output = args.output.resolve()
if args.output.exists():
    parser.error("--output must be a new file")
args.output.parent.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app

import gymnasium as gym
import torch

import insertanything_angle
from insertanything_angle.math import quat_rotate, axis_depth_gate
from InsertAnything.tasks.direct.insertanything.insertanything_env import InsertAnythingEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse


report = {
    "schema": "insertanything_angle_physical_contract_v1", "status": "running",
    "task": args.task, "seed": args.seed, "num_envs": args.num_envs,
    "steps": args.steps, "checks": {}, "policy_actions": "zero_action_hold",
}


def write_report():
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def check(condition, name, detail=None):
    passed = bool(condition)
    report["checks"][name] = {"passed": passed}
    if detail is not None:
        report["checks"][name]["detail"] = detail
    if not passed:
        raise AssertionError(f"Contract check failed: {name}: {detail}")


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def rng_snapshot():
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def rng_restore(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_equal(a, b):
    return (torch.equal(a["cpu"], b["cpu"]) and len(a["cuda"]) == len(b["cuda"])
            and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"])))


def stats(tensor):
    value = tensor.detach().float().cpu().flatten()
    return {"mean": value.mean().item(), "std": value.std(unbiased=False).item(),
            "min": value.min().item(), "max": value.max().item()}


def check_registration_and_config():
    entries = []
    for suffix, _, _ in insertanything_angle._TASKS:
        original = parse_env_cfg(f"InsertAnything-{suffix}-Direct-v0", device=args.device or "cuda:0", num_envs=args.num_envs)
        variant = parse_env_cfg(f"InsertAnythingAngle-{suffix}-Direct-v0", device=args.device or "cuda:0", num_envs=args.num_envs)
        old = serial(original.to_dict())
        new = serial(variant.to_dict())
        del new["angle_obs"], new["axis_alignment"], new["depth_axis_gate"]
        new["initial_in_hand_tilt_deg"] = old["initial_in_hand_tilt_deg"]
        new["ctrl"]["lock_roll_pitch"] = old["ctrl"]["lock_roll_pitch"]
        changed = [key for key in old.keys() | new.keys() if old.get(key) != new.get(key)]
        check(not changed, f"config_inherits_original_{suffix}", {"unexpected_fields": changed})
        check(variant.initial_in_hand_tilt_deg == 10 and not variant.ctrl.lock_roll_pitch,
              f"config_initialization_{suffix}")
        check(not variant.depth_axis_gate.enabled, f"legacy_depth_gate_disabled_{suffix}")
        # Verify mutable nested controls/tasks are isolated from original cfgs.
        variant.ctrl.default_task_prop_gains[0] += 1
        variant.task.hand_init_pos[0] += .123
        check(serial(original.to_dict()) == old, f"config_not_shared_{suffix}")
        entries.append({"task": f"InsertAnythingAngle-{suffix}-Direct-v0",
                        "robot_usd": original.robot.spawn.usd_path,
                        "nominal_episode_s": original.episode_length_s})
    report["registrations"] = entries


def check_observation_contract(b):
    before = rng_snapshot()
    noise_before = b._angle_noise.sample_count.clone()
    b._angle_noise.invalidate()
    original = InsertAnythingEnv._get_observations(b)
    after_base = rng_snapshot()
    rng_restore(before)
    expanded = b._get_observations()
    after_angle = rng_snapshot()
    check(rng_equal(after_base, after_angle), "angle_noise_does_not_consume_base_global_rng")
    check(torch.equal(expanded["policy"][:, :b.base_policy_dim], original["policy"]), "actor_prefix_exact")
    check(torch.equal(expanded["critic"][:, :b.base_critic_dim], original["critic"]), "critic_prefix_exact")
    check(expanded["policy"].shape == (b.num_envs, b.base_policy_dim + 1), "actor_space_shape")
    check(expanded["critic"].shape == (b.num_envs, b.base_critic_dim + 1), "critic_space_shape")
    if args.task == "InsertAnythingAngle-LHole-III-Direct-v0":
        check(expanded["policy"].shape[1] == 27 and expanded["critic"].shape[1] == 74, "L_task_27_74_observations")
    check(torch.equal(b._angle_noise.sample_count, noise_before + 1), "one_noise_draw_per_invalidated_env")
    first_tail = expanded["policy"][:, -1].clone()
    first_count = b._angle_noise.sample_count.clone()
    first_measurement = b.get_angle_diagnostics()["measured_in_hand_angle_rad"].clone()
    second = b._get_observations()
    second_diag = b.get_angle_diagnostics()
    check(torch.equal(first_tail, second["policy"][:, -1])
          and torch.equal(first_measurement, second_diag["measured_in_hand_angle_rad"])
          and torch.equal(first_count, b._angle_noise.sample_count), "repeated_reads_share_angle_noise")
    scale = math.radians(b.cfg.angle_obs.scale_deg)
    check(torch.equal(second["policy"][:, -1], second_diag["measured_in_hand_angle_rad"] / scale), "actor_angle_normalization")
    check(torch.equal(second["critic"][:, -1], second_diag["true_in_hand_angle_rad"] / scale), "critic_true_angle_normalization")
    # Inspection consumes the same base RNG budget as one original obs call.
    rng_restore(after_base)
    report["observation_shapes"] = {k: list(v.shape) for k, v in expanded.items()}


def check_reward_contract(b):
    attrs = ("prev_actions", "ep_succeeded", "ep_success_times")
    saved = {name: getattr(b, name).clone() for name in attrs}
    saved_extras = copy.deepcopy(b.extras)
    saved_diagnostics = b.angle_diagnostics
    weight = b.cfg.axis_alignment.reward_weight
    gate_enabled = b.cfg.depth_axis_gate.enabled

    def restore():
        for name, value in saved.items():
            setattr(b, name, value.clone())
        b.extras = copy.deepcopy(saved_extras)

    try:
        base = InsertAnythingEnv._get_rewards(b).clone()
        restore()
        total = b._get_rewards().clone()
        diag = {k: v.clone() for k, v in b.angle_diagnostics.items()}
        check(torch.equal(base, diag["original_reward"]), "base_reward_exact")
        expected_total = base + diag["alignment_reward"]
        if gate_enabled:
            expected_total = expected_total + diag["depth_gate_reward_adjustment"]
        check(torch.equal(total, expected_total), "reward_composition_exact")
        check(b.cfg.axis_alignment.zero_at_alignment, "axis_penalty_zero_at_alignment_enabled")
        check((diag["alignment_reward"] <= 0).all().item()
              and (diag["alignment_reward"] >= -weight).all().item(), "axis_penalty_in_negative_weight_to_zero")
        expected = weight * (torch.exp(-torch.square(diag["axis_error_rad"] / math.radians(b.cfg.axis_alignment.sigma_deg))) - 1.0)
        check(torch.allclose(diag["alignment_reward"], expected, atol=2e-7, rtol=1e-6),
              "axis_penalty_matches_requested_exp_minus_one")
        restore()
        b.cfg.axis_alignment.reward_weight = 0.0
        zero = b._get_rewards()
        expected_zero = base + diag["depth_gate_reward_adjustment"] if gate_enabled else base
        check(torch.equal(zero, expected_zero), "zero_axis_weight_retains_depth_gate")
        check(torch.count_nonzero(b.angle_diagnostics["alignment_reward"]).item() == 0, "zero_weight_removes_only_alignment_term")
        # Exercise gate on/off on exactly the same physical state and previous action.
        for enabled in (False, True):
            restore()
            b.cfg.axis_alignment.reward_weight = weight
            b.cfg.depth_axis_gate.enabled = enabled
            observed = b._get_rewards()
            current = b.angle_diagnostics
            gc = b.cfg.depth_axis_gate
            expected_gate = (axis_depth_gate(current["axis_error_rad"], gc.tolerance_deg, gc.sigma_deg, gc.min_gate)
                             if enabled else torch.ones_like(base))
            expected_depth = current["depth_reward_before_axis_gate"] * expected_gate
            expected_delta = expected_depth - current["depth_reward_before_axis_gate"]
            expected = base + current["alignment_reward"]
            if enabled:
                expected += expected_delta
            check(torch.allclose(observed, expected, atol=2e-6, rtol=0), f"gate_{enabled}_changes_only_weighted_depth")
            check(torch.equal(expected_gate, current["depth_axis_gate"]), f"gate_{enabled}_true_geometry")
            check(torch.equal(expected_depth, current["depth_reward_after_axis_gate"]), f"gate_{enabled}_logged_depth")
            check(torch.equal(base, current["original_reward"]), f"gate_{enabled}_original_reward_preserved")
            check(torch.equal(b.prev_actions, b.actions), f"gate_{enabled}_previous_action_updated")
            if not enabled:
                check(torch.equal(observed, base + current["alignment_reward"]), "disabled_gate_exact_legacy_reward")
            else:
                check((current["depth_gate_reward_adjustment"] <= 0).all().item(), "gate_never_adds_depth_reward")
                mask = current["axis_error_deg"] <= gc.tolerance_deg
                check(torch.equal(current["depth_axis_gate"][mask], torch.ones_like(base[mask])), "full_depth_reward_inside_tolerance")
        restore()
        b.cfg.depth_axis_gate.enabled = False
        b.cfg.axis_alignment.reward_weight = 0.
        check(torch.equal(b._get_rewards(), base), "both_disabled_exact_original_reward")
        report["initial_reward"] = {"original": stats(base), "alignment": stats(diag["alignment_reward"]),
                                   "depth_gate": stats(diag["depth_axis_gate"]),
                                   "depth_adjustment": stats(diag["depth_gate_reward_adjustment"]), "total": stats(total)}
    finally:
        b.cfg.axis_alignment.reward_weight = weight
        b.cfg.depth_axis_gate.enabled = gate_enabled
        restore()
        b.angle_diagnostics = saved_diagnostics


def main():
    torch.set_num_threads(4)
    check_registration_and_config()
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.depth_axis_gate.enabled = args.depth_axis_gate
    report["depth_axis_gate"] = cfg.depth_axis_gate.to_dict()
    env = gym.make(args.task, cfg=cfg)
    b = env.unwrapped
    try:
        check(not b.cfg.ctrl.lock_roll_pitch, "six_dof_actions_enabled")
        check(b.cfg.action_space == 6, "six_action_dimensions")
        check(type(b)._setup_scene is InsertAnythingEnv._setup_scene, "scene_implementation_inherited")
        check(type(b)._get_curr_successes is InsertAnythingEnv._get_curr_successes, "native_success_inherited")
        check(type(b)._get_dones is InsertAnythingEnv._get_dones, "termination_inherited")
        check(type(b)._apply_action is InsertAnythingEnv._apply_action, "action_controller_inherited")
        names = list(b._robot.body_names)
        report["robot_body_names"] = names
        check(not any(any(token in name.lower() for token in ("gel", "tactile", "digit")) for name in names), "no_tactile_robot_bodies")
        obs, _ = env.reset(seed=args.seed)
        diag = b.get_angle_diagnostics()
        local_z = torch.zeros((b.num_envs, 3), device=b.device)
        local_z[:, 2] = 1
        tcp_axis_w = quat_rotate(b.fingertip_midpoint_quat, local_z)
        down = -local_z
        upright_error = torch.rad2deg(torch.atan2(
            torch.linalg.vector_norm(torch.linalg.cross(tcp_axis_w, down, dim=-1), dim=-1),
            (tcp_axis_w * down).sum(dim=-1)))
        actual_axis_tcp = quat_apply_inverse(b.fingertip_midpoint_quat, quat_apply(b.held_quat, local_z))
        independently_measured = torch.rad2deg(torch.atan2(-actual_axis_tcp[:, 0], -actual_axis_tcp[:, 2]))
        check(torch.allclose(independently_measured, diag["true_in_hand_angle_deg"], atol=1e-4, rtol=0), "angle_uses_actual_physical_orientation")
        sampled = torch.rad2deg(b.initial_in_hand_tilt)
        report["initial_geometry_deg"] = {
            "tcp_from_vertical_down": stats(upright_error),
            "sampled_in_hand_tilt": stats(sampled),
            "actual_in_hand_angle": stats(diag["true_in_hand_angle_deg"]),
            "actual_minus_sampled": stats(diag["true_in_hand_angle_deg"] - sampled),
            "axis_error": stats(diag["axis_error_deg"]),
            "off_axis_residual": stats(diag["off_axis_residual_deg"]),
        }
        check(upright_error.max().item() < 2.0, "initial_tcp_vertical_within_2deg", report["initial_geometry_deg"]["tcp_from_vertical_down"])
        check(sampled.abs().max().item() <= b.cfg.initial_in_hand_tilt_deg + 1e-4, "sampled_tilt_within_configured_range")
        check_observation_contract(b)
        check_reward_contract(b)
        observed_noise = []
        trajectory = []
        actions = torch.zeros((b.num_envs, 6), device=b.device)
        for step in range(args.steps):
            counts = b._angle_noise.sample_count.clone()
            obs, reward, terminated, truncated, _ = env.step(actions)
            diag = b.angle_diagnostics
            check(all(torch.isfinite(value).all().item() for value in obs.values())
                  and torch.isfinite(reward).all().item() and torch.isfinite(b.actions).all().item(), f"finite_step_{step + 1}")
            check(not (terminated | truncated).any().item(), f"no_unexpected_termination_step_{step + 1}")
            check(torch.equal(b._angle_noise.sample_count, counts + 1), f"one_noise_sample_step_{step + 1}")
            expected_reward = diag["original_reward"] + diag["alignment_reward"]
            if cfg.depth_axis_gate.enabled:
                expected_reward += diag["depth_gate_reward_adjustment"]
            check(torch.equal(reward, expected_reward), f"composed_reward_step_{step + 1}")
            check((diag["alignment_reward"] <= 0).all().item(), f"nonpositive_axis_penalty_step_{step + 1}")
            observed_noise.append(diag["measurement_noise_deg"].detach().cpu().clone())
            trajectory.append({"step": step + 1, "reward_mean": reward.mean().item(),
                               "axis_error_deg": stats(diag["axis_error_deg"]),
                               "true_in_hand_angle_deg": stats(diag["true_in_hand_angle_deg"])})
        noise = torch.stack(observed_noise)
        report["observed_noise_deg"] = {**stats(noise), "sample_count": noise.numel(),
                                        "configured_std": b.cfg.angle_obs.noise_std_deg}
        configured_std = float(b.cfg.angle_obs.noise_std_deg)
        if configured_std == 0.0:
            check(torch.count_nonzero(noise).item() == 0,
                  "angle_noise_exactly_zero", report["observed_noise_deg"])
            scale = math.radians(b.cfg.angle_obs.scale_deg)
            check(torch.equal(obs["policy"][:, -1], diag["true_in_hand_angle_rad"] / scale),
                  "actor_angle_equals_actual_angle_without_noise")
        else:
            if noise.numel() >= 256:
                check(abs(noise.mean().item()) < .25 * configured_std
                      and .7 * configured_std < noise.std(unbiased=False).item() < 1.3 * configured_std,
                      "empirical_configured_gaussian_noise", report["observed_noise_deg"])
            if b.num_envs > 1:
                check(torch.unique(noise[0]).numel() > 1, "noise_independent_across_envs")
        report["trajectory_summary"] = trajectory
        # Exercise the actual inherited physical partial reset after rollout.
        before_count = b._angle_noise.sample_count.clone()
        ids = torch.tensor([0], device=b.device, dtype=torch.long)
        b._reset_idx(ids)
        check(not b._angle_noise.valid[0].item() and b._angle_noise.valid[1:].all().item(), "physical_partial_reset_invalidates_only_reset_env")
        reset_obs = b._get_observations()
        expected_count = before_count.clone()
        expected_count[0] += 1
        check(torch.equal(b._angle_noise.sample_count, expected_count), "physical_partial_reset_samples_only_reset_env")
        check(all(torch.isfinite(value).all().item() for value in reset_obs.values()), "partial_reset_observations_finite")
        report["status"] = "passed"
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
        write_report()
        print("ANGLE_CONTRACT_REPORT " + str(args.output), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        write_report()
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
