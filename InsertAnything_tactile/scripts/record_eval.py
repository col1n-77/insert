"""Evaluate one pretrained-policy episode and save synchronized stereo tactile data."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="InsertAnythingTactile-LHole-III-Direct-v0")
parser.add_argument("--num_envs", type=int, choices=(1,), default=1,
                    help="This synchronized video recorder records one environment and one full episode.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tactile_calibration", choices=("geometry_per_side", "legacy_shared"), default="geometry_per_side")
parser.add_argument("--checkpoint", type=Path, default=None,
                    help="Defaults to the original checkpoint matching --task (available for L/Hexagon III).")
parser.add_argument("--output", type=Path, default=None, help="New directory; existing outputs are never overwritten.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.task.startswith("InsertAnythingTactile-"):
    parser.error("Use an independent InsertAnythingTactile task ID")
if args.checkpoint is None:
    original = Path(os.environ.get("INSERTANYTHING_ROOT", str(ROOT.parent)))
    args.checkpoint = (original / "source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints"
                       / (args.task.replace("InsertAnythingTactile-", "InsertAnything-", 1) + ".pth"))
if not args.checkpoint.is_file():
    parser.error(f"No trained policy at {args.checkpoint}; supply a compatible --checkpoint")
args.checkpoint = args.checkpoint.resolve()
output = (args.output or ROOT / "outputs" / datetime.now().strftime("eval_%Y%m%d_%H%M%S_%f")).resolve()
if output.exists():
    parser.error(f"Output already exists: {output}")
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
import insertanything_tactile  # task registry
from insertanything_tactile.policy import CheckpointPolicy
from insertanything_tactile.recording import EpisodeRecorder, RecordingEnv
from isaaclab_tasks.utils import parse_env_cfg


def main():
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "in_progress", "task": args.task, "seed": args.seed, "num_envs": 1,
              "checkpoint": str(args.checkpoint), "sensor": "GelSight Mini", "optics": "Taxim",
              "mechanics": "PhysX rigid gelpads", "uses_tactile_for_actions": False,
              "output": str(output)}
    with args.checkpoint.open("rb") as file:
        report["checkpoint_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"RECORD_OUTPUT {output}", flush=True)
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1)
    cfg.seed = args.seed
    cfg.evaluation_mode = True
    cfg.tactile.sides = ("left", "right")
    cfg.tactile.calibration_mode = args.tactile_calibration
    cfg.viewer.resolution = (960, 720)
    env = None
    recorder = None
    try:
        env = RecordingEnv(cfg, render_mode="rgb_array")
        with torch.inference_mode():
            policy = CheckpointPolicy(env, args.task, args.checkpoint, args.seed)
            observations, _ = env.reset()
            # Create and warm up the scene annotator without advancing physics.
            env.render()
            for _ in range(4):
                env.sim.render()
            tactile = env.refresh_tactile()
            np.savez_compressed(output / "reset.npz", **{
                f"{side}_{key}": value.detach().cpu().numpy()
                for side, values in tactile.items() for key, value in values.items()
            })
            report.update({
                "physics_dt_s": env.physics_dt, "control_dt_s": env.step_dt,
                "sampling_hz": 1.0 / env.step_dt, "max_episode_steps": env.max_episode_length,
                "observation_shapes": {key: list(value.shape) for key, value in observations.items()},
                "policy_deterministic": bool(policy.agent.is_deterministic),
                "robot_joint_names": env._robot.joint_names,
                "joint_position_units": ["rad"] * 7 + ["m"] * 2,
                "joint_velocity_units": ["rad/s"] * 7 + ["m/s"] * 2,
                "commanded_joint_effort_units": ["N*m"] * 7 + ["N"] * 2,
                "scene_resolution_wh": list(cfg.viewer.resolution),
                "tactile_resolution_wh": list(cfg.tactile.resolution),
                "env_origin_world_m": env.scene.env_origins[0].cpu().tolist(),
                "contact_force_config": cfg.task.contact_force,
                "data_notes": {
                    "contact_local_n": "Upstream peg-frame contact-force signal: tare and EMA processed; N.",
                    "net_normal_world_n": "Sum of peg contact sensor net NORMAL forces in world axes; excludes tangential forces and is not pair-filtered.",
                    "joint_effort": "Controller command, not a measured joint torque or tactile force.",
                    "press_depth_mm": "Geometric depth-to-Taxim indentation proxy, not pressure or a calibrated force.",
                    "time_s": "Time since policy rollout start; excludes grasp initialization/reset physics.",
                    "videos": "Lossy H.264 previews; lossless tactile RGB and float32 depth are in trajectory.h5.",
                },
            })
            recorder = EpisodeRecorder(output, env, report)
            env.recorder = recorder
            for _ in range(env.max_episode_length + 1):
                if not app.is_running():
                    raise RuntimeError("Simulation was closed before the episode completed")
                actions = policy.act(observations)
                recorder.before_action(observations, actions, policy.clip_obs)
                observations, reward, terminated, truncated, info = env.step(actions)
                if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in observations.values()):
                    raise RuntimeError("Non-finite rollout output")
                if bool((terminated | truncated).any()):
                    if "eval_printout" in info:
                        print(info["eval_printout"], end="", flush=True)
                        report["episode_summary"] = info["eval_printout"]
                    break
            env.recorder = None
            report.update(recorder.finish())
            recorder.close()
            recorder = None
            report["status"] = "complete"
            report["files"] = ["scene.mp4", "left_tactile.mp4", "right_tactile.mp4",
                               "trajectory.h5", "states.csv", "reset.npz"]
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("RECORD_REPORT " + json.dumps(report), flush=True)
            print(f"OUTPUT {output}", flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        if env is not None:
            env.recorder = None
        try:
            if recorder is not None:
                recorder.close()
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
