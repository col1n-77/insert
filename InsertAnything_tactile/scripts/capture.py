"""Finite tactile rollout; saves raw RGB/depth and a contact-removal check."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="InsertAnythingTactile-LHole-III-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sensors", choices=("left", "right", "both"), default="both")
parser.add_argument("--tactile_calibration", choices=("geometry_per_side", "legacy_shared"), default="geometry_per_side")
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--check_contact", action="store_true")
parser.add_argument("--checkpoint", type=Path, default=None,
                    help="Optional original RL-Games checkpoint; otherwise hold the initialized grasp.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.num_envs < 1:
    parser.error("steps and num_envs must be positive")
if args.checkpoint and not args.checkpoint.is_file():
    parser.error(f"checkpoint does not exist: {args.checkpoint}")
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import insertanything_tactile
from isaaclab_tasks.utils import parse_env_cfg


def save_frame(output, name, data):
    arrays = {}
    for side, values in data.items():
        for key, value in values.items():
            arrays[f"{side}_{key}"] = value.detach().cpu().numpy()
        Image.fromarray(arrays[f"{side}_rgb"][0]).save(output / f"{name}_{side}.png")
    np.savez_compressed(output / f"{name}.npz", **arrays)


def main():
    output = args.output or Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.evaluation_mode = True
    cfg.tactile.sides = ("left", "right") if args.sensors == "both" else (args.sensors,)
    cfg.tactile.calibration_mode = args.tactile_calibration
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
    base = env.unwrapped
    report = {"task": args.task, "num_envs": args.num_envs, "steps": args.steps,
              "seed": args.seed, "sensor": "GelSight Mini", "optics": "Taxim",
              "mechanics": "PhysX rigid gelpads", "uses_tactile_for_actions": False,
              "tactile_calibration": base.tactile_calibration}
    try:
        with torch.inference_mode():
            policy = None
            if args.checkpoint:
                from insertanything_tactile.policy import CheckpointPolicy
                policy = CheckpointPolicy(env, args.task, args.checkpoint, args.seed)
                report["checkpoint"] = str(args.checkpoint.resolve())
            report["action_source"] = "checkpoint" if policy else "zero_action_hold"
            obs, _ = env.reset()
            base.refresh_tactile()
            save_frame(output, "reset", base.get_tactile_observations())
            depth_max = {side: 0.0 for side in cfg.tactile.sides}
            rgb_first = {side: data["rgb"].clone() for side, data in base.get_tactile_observations().items()}
            completed_trials = 0
            episode_summaries = []
            for step in range(args.steps):
                actions = policy.act(obs) if policy else torch.zeros_like(base.actions)
                if not torch.isfinite(actions).all():
                    raise RuntimeError(f"Non-finite actions at step {step}")
                obs, reward, terminated, truncated, info = env.step(actions)
                dones = terminated | truncated
                if dones.any():
                    completed_trials += int(dones.sum())
                    if policy:
                        policy.reset_done(dones)
                if "eval_printout" in info:
                    print(info["eval_printout"], end="", flush=True)
                    episode_summaries.append(info["eval_printout"])
                if not torch.isfinite(reward).all() or not all(torch.isfinite(value).all() for value in obs.values()):
                    raise RuntimeError(f"Non-finite rollout at step {step}")
                data = base.get_tactile_observations()
                for side, values in data.items():
                    width, height = cfg.tactile.resolution
                    assert tuple(values["rgb"].shape) == (args.num_envs, height, width, 3)
                    assert values["rgb"].dtype == torch.uint8
                    assert torch.isfinite(values["press_depth_mm"]).all()
                    depth_max[side] = max(depth_max[side], float(values["press_depth_mm"].max()))
                if (step + 1) % 30 == 0:
                    print(f"TACTILE_PROGRESS {step + 1}/{args.steps}", flush=True)
            base.refresh_tactile()
            contact = base.get_tactile_observations(clone=True)
            save_frame(output, "contact", contact)
            env.render()
            for _ in range(4):
                base.sim.render()
            frame = env.render()
            if frame is not None and frame.size:
                Image.fromarray(frame).save(output / "scene.png")
            report["observation_shapes"] = {key: list(value.shape) for key, value in obs.items()}
            report["completed_trials"] = completed_trials
            report["episode_summaries"] = episode_summaries
            report["max_press_depth_mm"] = depth_max
            report["rgb_change_from_reset_mean"] = {
                side: float((values["rgb"].float() - rgb_first[side].float()).abs().mean())
                for side, values in contact.items()
            }
            report["robot_body_names"] = base._robot.body_names
            if args.check_contact:
                # Render the same gripper pose with only the peg removed. This
                # rules out a static gel/housing/background image as contact.
                saved_state = base._held_asset.data.root_state_w.clone()
                away = saved_state[:, :7].clone()
                away[:, 0] += 1.0
                base._held_asset.write_root_pose_to_sim(away)
                base._held_asset.write_root_velocity_to_sim(torch.zeros_like(saved_state[:, 7:]))
                for _ in range(4):
                    base.step_sim_no_action()
                base.refresh_tactile()
                removed = base.get_tactile_observations(clone=True)
                save_frame(output, "peg_removed", removed)
                report["contact_check"] = {}
                for side in cfg.tactile.sides:
                    change = (contact[side]["rgb"].float() - removed[side]["rgb"].float()).abs().mean((1, 2, 3))
                    on = contact[side]["press_depth_mm"].amax((1, 2))
                    off = removed[side]["press_depth_mm"].amax((1, 2))
                    report["contact_check"][side] = {
                        "contact_max_mm": on.tolist(), "removed_max_mm": off.tolist(),
                        "rgb_mean_abs_change": change.tolist(),
                        "passed": bool(((on > off + 0.01) & (change > 0.01)).all()),
                    }
                base._held_asset.write_root_state_to_sim(saved_state)
                base.step_sim_no_action()
                base.refresh_tactile()
            report["status"] = "passed"
            if args.check_contact and not all(item["passed"] for item in report["contact_check"].values()):
                report["status"] = "contact_check_failed"
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("TACTILE_REPORT " + json.dumps(report), flush=True)
            print(f"OUTPUT {output.resolve()}", flush=True)
            if report["status"] != "passed":
                raise RuntimeError("Contact image validation failed; inspect report.json")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        # Kit's native fast shutdown may exit 0 or hang while unwinding an
        # exception. Preserve the failure status after main's env.close().
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
