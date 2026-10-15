"""Bounded dynamic integration test; logs estimates/truth separately and saves RGB."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--task", default="InsertAnythingTactile6D-LHole-III-Direct-v0")
p.add_argument("--angle_checkpoint", type=Path, default=Path("outputs/dynamic_angle_rl_model_20261012/best.pt"))
p.add_argument("--policy_checkpoint", type=Path, default=None, help="Optional new six-DOF RL policy for a finite evaluation")
p.add_argument("--num_envs", type=int, default=2)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--seed", type=int, default=31)
p.add_argument("--initial_tilt_deg", type=float, default=10., help="Peg tilt about TCP +Y; wrist starts upright")
p.add_argument("--episode_steps", type=int, default=None, help="Optional short episode override; default uses task duration")
p.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.enable_cameras = True
if min(args.steps, args.num_envs) < 1 or (args.episode_steps is not None and args.episode_steps < 1): p.error("Counts must be positive")
app = AppLauncher(args).app
import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import insertanything_tactile.rl
from isaaclab_tasks.utils import parse_env_cfg
from tactile_pose.rl_math import quat_mul, quat_inverse, local_z_w


def reset_snapshot(b):
    """Measure actual poses after physical closing, not only configured targets."""
    diag = b.angle_diagnostics()
    tcp_tilt = torch.rad2deg(torch.acos((-local_z_w(b.fingertip_midpoint_quat)[:, 2]).clamp(-1, 1)))
    sampled = torch.rad2deg(b.initial_in_hand_tilt)
    assert b.cfg_task.hand_init_orn_range[:2] == [0.0, 0.0]
    assert tcp_tilt.max() < 1.0, "Wrist did not initialize upright"
    assert sampled.abs().max() <= b.cfg.tactile_alignment.initial_tilt_deg + 1e-4
    assert (diag["gt_in_hand_angle_deg"] - sampled).abs().max() < 3.0, "Physical grasp lost the sampled peg tilt"
    return {
        "sampled_in_hand_deg": sampled.tolist(),
        "tcp_world_tilt_deg": tcp_tilt.tolist(),
        "actual_in_hand_deg": diag["gt_in_hand_angle_deg"].tolist(),
        "peg_hole_axis_error_deg": diag["gt_axis_error_deg"].tolist(),
        "closure_angle_drift_deg": (diag["gt_in_hand_angle_deg"] - sampled).tolist(),
        "predicted_angle_deg": diag["predicted_angle_deg"].tolist(),
        "prediction_error_deg": diag["gt_prediction_abs_error_deg"].tolist(),
        "valid_contact": diag["valid_fraction"].tolist(),
        "tcp_quat_wxyz": b.fingertip_midpoint_quat.tolist(),
        "object_quat_wxyz": b.held_quat.tolist(),
    }


def main():
    output = args.output or Path("outputs")/datetime.now().strftime("tactile_rl_check_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.tactile_alignment.checkpoint = str(args.angle_checkpoint.resolve())
    cfg.tactile_alignment.initial_tilt_deg = args.initial_tilt_deg
    if args.episode_steps: cfg.episode_length_s = args.episode_steps*cfg.decimation*cfg.sim.dt
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
    b = env.unwrapped
    rows, resets = [], 0
    try:
        with torch.inference_mode():
            policy = None
            if args.policy_checkpoint:
                previous = json.loads((args.policy_checkpoint.resolve().parent.parent/"run.json").read_text())
                if (previous.get("schema") != "insertanything_tactile6d_rl_v2" or previous["task"] != args.task
                        or previous.get("initialization") != b.initialization_metadata
                        or previous["angle_estimator"]["sha256"] != b.angle_estimator.metadata["sha256"]):
                    raise ValueError("Policy/task/initialization/angle-model contract mismatch")
                from insertanything_tactile.policy import CheckpointPolicy
                policy = CheckpointPolicy(env, args.task, args.policy_checkpoint, args.seed)
                policy.agent.is_deterministic = True
            obs, _ = env.reset()
            initial_diag = {k: v.tolist() for k,v in b.angle_diagnostics().items()}
            initializations = [reset_snapshot(b)]
            for side, data in b.get_tactile_observations().items():
                Image.fromarray(data["rgb"][0].cpu().numpy()).save(output/f"reset_{side}.png")
            before = b.angle_estimator.calls
            b._get_observations(); b._get_curr_successes(b.cfg_task.success_threshold)
            assert b.angle_estimator.calls == before, "Repeated reads inferred from stale/new images inconsistently"
            target_rotations = []
            actual_rotations = []
            estimated_successes = ground_truth_successes = 0
            episode_est = torch.zeros(b.num_envs, dtype=torch.bool, device=b.device)
            episode_gt = torch.zeros_like(episode_est)
            for step in range(args.steps):
                previous_q = b.fingertip_midpoint_quat.clone()
                actions = torch.zeros_like(b.actions)
                period = args.episode_steps or args.steps
                phase = step % period
                actions[:, 3 if phase < period//2 else 4] = .6
                if policy is not None: actions = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(actions)
                assert torch.isfinite(reward).all() and all(torch.isfinite(x).all() for x in obs.values())
                assert obs["policy"].shape == (args.num_envs, cfg.observation_space)
                assert obs["critic"].shape == (args.num_envs, cfg.state_space)
                delta = quat_mul(b.rl_control_target_quat, quat_inverse(previous_q))
                target_rotations.append(delta[:, 1:].abs().max(0).values.cpu().numpy())
                dones = terminated | truncated
                diag = b.last_reward_diagnostics
                episode_est |= diag["estimated_insert_success"]
                episode_gt |= diag["gt_insert_success"]
                if not dones.any():
                    actual_delta = quat_mul(b.fingertip_midpoint_quat, quat_inverse(previous_q))
                    actual_rotations.append((2*torch.rad2deg(actual_delta[:, 1:].abs())).max(0).values.cpu().numpy())
                if dones.any():
                    assert dones.all(), "Expected synchronous episode resets"
                    resets += 1
                    estimated_successes += int(episode_est[dones].sum())
                    ground_truth_successes += int(episode_gt[dones].sum())
                    episode_est[dones] = False; episode_gt[dones] = False
                    if policy is not None: policy.reset_done(dones)
                    initializations.append(reset_snapshot(b))
                for e in range(args.num_envs):
                    rows.append({"step": step+1, "env_id": e, "post_reset_observation": bool(dones[e]),
                                 "diagnostic_phase": "post_action_pre_reset",
                                 "reward_before_reset": float(reward[e]),
                                 **{k: float(v[e]) for k,v in diag.items()}})
                if (step+1)%10 == 0: print(f"TACTILE_RL_CHECK {step+1}/{args.steps}", flush=True)
            for side, data in b.get_tactile_observations().items():
                Image.fromarray(data["rgb"][0].cpu().numpy()).save(output/f"final_{side}.png")
            # Force invalid contact without changing true poses to test reward masking.
            saved_valid = b.tactile_valid.clone(); b.tactile_valid[:] = False
            terms, _, _ = b._get_factory_rew_dict()
            assert not terms["tactile_axis_align"].any() and not terms["z_insert"].any()
            b.tactile_valid[:] = saved_valid
            target_rotations = np.stack(target_rotations)
            if policy is None:
                assert target_rotations[:, 0].max() > 1e-4 and target_rotations[:, 1].max() > 1e-4
                assert np.stack(actual_rotations)[:, :2].max(0).min() > .005
            valid = [r for r in rows if r["valid_fraction"]]
            errors = np.array([r["gt_prediction_abs_error_deg"] for r in valid])
            report = {"status": "policy_rollout_passed" if policy else "integration_passed", "task": args.task, "steps": args.steps,
                      "num_envs": args.num_envs, "seed": args.seed, "completed_batch_resets": resets,
                      "completed_env_episodes": resets*b.num_envs,
                      "estimated_success_episodes": estimated_successes,
                      "gt_success_episodes": ground_truth_successes,
                      "observation_shapes": {k: list(v.shape) for k,v in obs.items()},
                      "initial_diagnostics": initial_diag,
                      "initialization": b.initialization_metadata,
                      "reset_poses": initializations,
                      "action_source": str(args.policy_checkpoint) if policy else "roll_pitch_probe",
                      "roll_pitch_control_nonzero": bool(target_rotations[:, :2].max(0).min() > 1e-4),
                      "actual_angular_increment_max_deg_xyz": np.stack(actual_rotations).max(0).tolist() if actual_rotations else [],
                      "invalid_contact_reward_zero": True,
                      "estimator_calls": b.angle_estimator.calls, "estimator": b.angle_estimator.metadata,
                      "dynamic_valid_fraction": len(valid)/len(rows),
                      "dynamic_valid_angle_mae_deg": float(errors.mean()) if len(errors) else None,
                      "dynamic_valid_angle_p95_deg": float(np.percentile(errors,95)) if len(errors) else None,
                      "dynamic_gt_off_axis_mean_deg": float(np.mean([r["gt_off_axis_rotation_deg"] for r in rows])),
                      "note": "Integration test only; no policy convergence or insertion success-rate claim"}
            with (output/"diagnostics.csv").open("w") as f:
                writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
            print("TACTILE_RL_REPORT "+json.dumps(report), flush=True)
    finally: env.close()


if __name__ == "__main__":
    try: main()
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    app.close()
