"""Evaluate a saved tactile PPO policy on full episodes with held-out seeds.

The training env/agent configurations and angle-estimator contract are checked.
All step diagnostics are captured before Gym's automatic reset. No training or
environment reward/control changes are performed.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import yaml
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--policy_checkpoint", type=Path)
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003, 1004])
p.add_argument("--output", type=Path)
p.add_argument("--stochastic", action="store_true", help="Sample policy actions; default uses the mean action")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.policy_checkpoint is None or args.output is None:
    p.error("--policy_checkpoint and --output are required")
if args.num_envs < 1 or len(set(args.seeds)) != len(args.seeds):
    p.error("Use positive num_envs and distinct seeds")
checkpoint = args.policy_checkpoint.resolve(strict=True)
run_root = checkpoint.parent.parent
contract = json.loads((run_root / "run.json").read_text())
if contract.get("schema") != "insertanything_tactile6d_rl_v2":
    p.error("Expected a v2 tactile six-DOF policy")
if contract["seed"] in args.seeds:
    p.error("Evaluation seeds must differ from the training seed")
# FullLoader accepts the tuple tags written by Isaac Lab's dump_yaml.
saved_env = yaml.load((run_root / "params/env.yaml").read_text(), Loader=yaml.FullLoader)
saved_agent = yaml.safe_load((run_root / "params/agent.yaml").read_text())
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import insertanything_tactile.rl
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from insertanything_tactile.policy import CheckpointPolicy
from InsertAnything.tasks.direct.insertanything import insertanything_utils as utils
from tactile_pose.rl_math import local_z_w


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def wilson(k, n):
    z = 1.959963984540054
    centre = (k / n + z*z / (2*n)) / (1 + z*z/n)
    half = z * math.sqrt(k/n * (1-k/n)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0.0, centre-half), min(1.0, centre+half)]


def main():
    torch.set_num_threads(4)
    task = contract["task"]
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = int(saved_env["seed"])
    cfg.from_dict(saved_env)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seeds[0]
    if args.device:
        cfg.sim.device = args.device
    if cfg.episode_length_s != cfg.task.duration_s:
        raise ValueError("This evaluator requires training with the full task episode duration")
    # The reusable player loads the registry architecture; reject any mismatch
    # rather than silently evaluating with a different network or normalization.
    registry = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    for key in ("network", "model", "env", "algo"):
        if registry["params"][key] != saved_agent["params"][key]:
            raise ValueError(f"Player/training configuration mismatch: {key}")
    for key in ("normalize_input", "normalize_value", "clip_value"):
        if registry["params"]["config"][key] != saved_agent["params"]["config"][key]:
            raise ValueError(f"Player/training normalization mismatch: {key}")
    env = gym.make(task, cfg=cfg)
    b = env.unwrapped
    all_episodes, all_errors, all_valid, per_seed = [], [], [], []
    wall_start = time.monotonic()
    try:
        if b.initialization_metadata != contract["initialization"]:
            raise ValueError("Initialization contract mismatch")
        if b.angle_estimator.metadata["sha256"] != contract["angle_estimator"]["sha256"]:
            raise ValueError("Angle-estimator checkpoint mismatch")
        for key in ("observation_space", "state_space", "action_space"):
            if getattr(cfg, key) != contract[key]:
                raise ValueError(f"Policy/environment shape mismatch: {key}")
        policy = CheckpointPolicy(env, task, checkpoint, args.seeds[0])
        policy.agent.is_deterministic = not args.stochastic
        original_get_rewards = b._get_rewards
        current = {}

        def save_rgb(folder, name):
            for side, value in b.get_tactile_observations().items():
                Image.fromarray(value["rgb"][0].cpu().numpy()).save(folder / f"{name}_{side}.png")

        def capture_rewards():
            reward = original_get_rewards()
            diag = dict(b.last_reward_diagnostics)
            hp, hq = utils.get_held_base_pose(b.held_pos, b.held_quat, b.num_envs, b.device)
            tp, tq = utils.get_target_held_base_pose(b.fixed_pos, b.fixed_quat, b.cfg_task, b.num_envs, b.device)
            diag["xy_error_mm"] = torch.linalg.vector_norm(hp[:, :2]-tp[:, :2], dim=1)*1000
            diag["z_error_mm"] = (hp[:, 2]-tp[:, 2])*1000
            _, _, yaw = utils.get_closest_symmetry_transform(hq, tq, [math.radians(x) for x in b.cfg_task.symmetry_angles_deg])
            diag["yaw_error_deg"] = torch.rad2deg(yaw)
            diag["tcp_tilt_deg"] = torch.rad2deg(torch.acos((-local_z_w(b.fingertip_midpoint_quat)[:, 2]).clamp(-1, 1)))
            diag["opening_mm"] = b._robot.data.joint_pos[:, b._finger_joint_ids].sum(-1)*1000
            diag["contact_force_norm_n"] = torch.linalg.vector_norm(b.contact_force_local, dim=-1)
            diag["geometry_success_without_contact"] = ((diag["xy_error_mm"] < 2.5)
                & (diag["z_error_mm"] < b.cfg_task.fixed_asset_cfg.height*b.cfg_task.success_threshold*1000)
                & (diag["z_error_mm"] >= -2.0)
                & ((yaw < b.cfg_task.yaw_success_threshold) if b.cfg_task.requires_orientation_logic else torch.ones_like(yaw, dtype=torch.bool))
                & (diag["gt_axis_error_deg"] < cfg.tactile_alignment.success_angle_deg))
            expected_gt = diag["geometry_success_without_contact"] & diag["valid_fraction"]
            if not torch.equal(expected_gt, diag["gt_insert_success"]):
                raise RuntimeError("Independent geometry audit disagrees with task success")
            current["diag"] = {k: v.detach().cpu().numpy().copy() for k, v in diag.items()}
            current["state"] = {
                "tcp_pos_w": (b.fingertip_midpoint_pos+b.scene.env_origins).cpu().numpy().copy(),
                "tcp_quat_wxyz": b.fingertip_midpoint_quat.cpu().numpy().copy(),
                "object_pos_w": (b.held_pos+b.scene.env_origins).cpu().numpy().copy(),
                "object_quat_wxyz": b.held_quat.cpu().numpy().copy(),
                "hole_pos_w": (b.fixed_pos+b.scene.env_origins).cpu().numpy().copy(),
                "hole_quat_wxyz": b.fixed_quat.cpu().numpy().copy(),
                "joint_pos": b.joint_pos.cpu().numpy().copy(),
                "joint_vel": b.joint_vel.cpu().numpy().copy(),
                "contact_force_local_n": b.contact_force_local.cpu().numpy().copy(),
                "filtered_actions": b.actions.cpu().numpy().copy(),
            }
            if current["step"] in (1, b.max_episode_length//2, b.max_episode_length):
                save_rgb(current["folder"], f"step_{current['step']:04d}")
            return reward

        b._get_rewards = capture_rewards
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
        report_base = {
            "schema": "insertanything_tactile_full_eval_v1", "task": task,
            "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest(),
            "checkpoint_epoch": weights.get("epoch"), "training_frames": weights.get("frame"),
            "training_seed": contract["seed"], "num_envs": b.num_envs, "seeds": args.seeds,
            "deterministic": not args.stochastic, "episode_steps": b.max_episode_length,
            "episode_duration_s": cfg.episode_length_s, "step_dt_s": b.step_dt,
            "initialization": b.initialization_metadata, "angle_estimator": b.angle_estimator.metadata,
            "config_source": str(run_root / "params/env.yaml"),
            "diagnostic_phase": "post_action_pre_reset",
            "success_definition": "At least one step meeting original XY/depth/yaw criteria, depth >= -2 mm, true axis error <3 deg, and valid bilateral contact. Terminal and >=10 consecutive steps are reported separately.",
            "coordinates": "position world metres, quaternion wxyz; contact_force_local_n is the task contact wrench force, not calibrated gel pressure",
        }
        del weights
        dump(args.output / "report.json", {**report_base, "status": "running"})
        with torch.inference_mode():
            for seed in args.seeds:
                folder = args.output / f"seed_{seed}"
                folder.mkdir()
                current.update(folder=folder, step=0)
                obs, _ = env.reset(seed=seed)
                policy.agent.reset()
                policy.ready = False
                reset_data = {
                    "sampled_angle_deg": torch.rad2deg(b.initial_in_hand_tilt).tolist(),
                    "tcp_quat_wxyz": b.fingertip_midpoint_quat.tolist(),
                    "object_quat_wxyz": b.held_quat.tolist(),
                    "tcp_pos_w": (b.fingertip_midpoint_pos+b.scene.env_origins).tolist(),
                    "object_pos_w": (b.held_pos+b.scene.env_origins).tolist(),
                    "hole_pos_w": (b.fixed_pos+b.scene.env_origins).tolist(),
                    "diagnostics": {k: v.tolist() for k, v in b.angle_diagnostics().items()},
                }
                dump(folder / "reset.json", reset_data)
                save_rgb(folder, "reset")
                rows, state_frames = [], []
                seed_start = time.monotonic()
                with (folder / "diagnostics.csv").open("w") as file:
                    writer = None
                    for step in range(1, b.max_episode_length+1):
                        current["step"] = step
                        actions = policy.act(obs)
                        obs, reward, terminated, truncated, _ = env.step(actions)
                        if not torch.isfinite(actions).all() or not torch.isfinite(reward).all() or not all(torch.isfinite(x).all() for x in obs.values()):
                            raise RuntimeError("Non-finite action, reward, or observation")
                        dones = terminated | truncated
                        if bool(dones.any()) != (step == b.max_episode_length) or (dones.any() and not dones.all()):
                            raise RuntimeError("Unexpected episode length or partial reset")
                        diag = current["diag"]
                        state_frames.append({**current["state"], "raw_actions": actions.cpu().numpy().copy()})
                        rewards = reward.cpu().numpy()
                        for e in range(b.num_envs):
                            row = {"seed": seed, "env_id": e, "step": step, "time_s": step*b.step_dt,
                                   "reward": float(rewards[e]), **{k: float(v[e]) for k, v in diag.items()}}
                            if writer is None:
                                writer = csv.DictWriter(file, fieldnames=list(row)); writer.writeheader()
                            writer.writerow(row); rows.append(row)
                        if step % 50 == 0:
                            file.flush()
                            print(f"EVAL_PROGRESS seed={seed} step={step}/{b.max_episode_length} elapsed={time.monotonic()-seed_start:.1f}s", flush=True)
                np.savez_compressed(folder / "states.npz", **{k: np.stack([x[k] for x in state_frames]) for k in state_frames[0]})
                episode_results = []
                for e in range(b.num_envs):
                    trajectory = rows[e::b.num_envs]
                    successes = [bool(r["gt_insert_success"]) for r in trajectory]
                    run = longest = 0
                    for value in successes:
                        run = run+1 if value else 0; longest = max(longest, run)
                    valid_rows = [r for r in trajectory if r["valid_fraction"]]
                    episode_results.append({
                        "seed": seed, "env_id": e, "sampled_angle_deg": reset_data["sampled_angle_deg"][e],
                        "gt_success": any(successes), "estimated_success": any(r["estimated_insert_success"] for r in trajectory),
                        "geometry_success_without_contact": any(r["geometry_success_without_contact"] for r in trajectory),
                        "terminal_gt_success": successes[-1], "sustained_10steps_gt_success": longest >= 10,
                        "first_gt_success_s": next((r["time_s"] for r in trajectory if r["gt_insert_success"]), None),
                        "return": sum(r["reward"] for r in trajectory),
                        "valid_fraction": len(valid_rows)/len(trajectory),
                        "valid_angle_mae_deg": float(np.mean([r["gt_prediction_abs_error_deg"] for r in valid_rows])) if valid_rows else None,
                        "min_gt_axis_error_deg": min(r["gt_axis_error_deg"] for r in trajectory),
                        **{"terminal_"+k: trajectory[-1][k] for k in ("xy_error_mm", "z_error_mm", "yaw_error_deg", "gt_axis_error_deg", "estimated_axis_error_deg", "valid_fraction", "gt_off_axis_rotation_deg")},
                    })
                all_episodes.extend(episode_results)
                all_errors.extend(r["gt_prediction_abs_error_deg"] for r in rows if r["valid_fraction"])
                all_valid.extend(bool(r["valid_fraction"]) for r in rows)
                seed_result = {"seed": seed, "episodes": episode_results, "elapsed_s": time.monotonic()-seed_start}
                per_seed.append(seed_result)
                dump(folder / "episodes.json", seed_result)
                n = len(all_episodes); k = sum(e["gt_success"] for e in all_episodes)
                aggregate = {**report_base, "status": "completed" if seed == args.seeds[-1] else "running",
                    "completed_episodes": n, "gt_success_episodes": k, "gt_success_rate": k/n, "gt_success_wilson95": wilson(k,n),
                    "estimated_success_episodes": sum(e["estimated_success"] for e in all_episodes),
                    "terminal_gt_success_episodes": sum(e["terminal_gt_success"] for e in all_episodes),
                    "sustained_10steps_gt_success_episodes": sum(e["sustained_10steps_gt_success"] for e in all_episodes),
                    "geometry_success_without_contact_episodes": sum(e["geometry_success_without_contact"] for e in all_episodes),
                    "mean_return": float(np.mean([e["return"] for e in all_episodes])),
                    "contact_valid_fraction": float(np.mean(all_valid)),
                    "valid_angle_mae_deg": float(np.mean(all_errors)) if all_errors else None,
                    "valid_angle_p95_deg": float(np.percentile(all_errors,95)) if all_errors else None,
                    "mean_terminal_gt_axis_error_deg": float(np.mean([e["terminal_gt_axis_error_deg"] for e in all_episodes])),
                    "elapsed_s": time.monotonic()-wall_start, "per_seed": per_seed,
                }
                dump(args.output / "report.json", aggregate)
                print(f"EVAL_BATCH seed={seed} gt={sum(e['gt_success'] for e in episode_results)}/{b.num_envs} total={k}/{n}", flush=True)
        print("TACTILE_EVAL_OUTPUT " + str(args.output), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    app.close()
