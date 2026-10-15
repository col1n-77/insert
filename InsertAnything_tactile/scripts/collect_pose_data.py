"""Collect independent physical grasps for stereo tactile 3D rotation supervision."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="InsertAnythingTactile-LHole-III-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--grasps", type=int, default=24, help="Total independent grasps, divisible by num_envs")
parser.add_argument("--frames_per_grasp", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--rotation_range_deg", type=float, nargs=3, default=(20, 20, 35), metavar=("X", "Y", "Z"))
parser.add_argument("--contact_depth_mm", type=float, default=0.02)
parser.add_argument("--contact_min_pixels", type=int, default=64)
parser.add_argument("--tactile_calibration", choices=("geometry_per_side", "legacy_shared"), default="geometry_per_side")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.num_envs, args.grasps, args.frames_per_grasp, args.contact_min_pixels) < 1:
    parser.error("Counts must be positive")
if args.grasps % args.num_envs:
    parser.error("grasps must be divisible by num_envs")
if args.seed < 0 or min(args.rotation_range_deg) < 0 or args.contact_depth_mm <= 0:
    parser.error("Use non-negative seeds/ranges and positive contact threshold")
output = (args.output or ROOT / "outputs" / datetime.now().strftime("pose_%Y%m%d_%H%M%S_%f")).resolve()
if output.exists():
    parser.error(f"Output already exists: {output}")
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import insertanything_tactile
from isaaclab_tasks.utils import parse_env_cfg
from tactile_pose.sim_collection import PoseCollectionEnv, PoseWriter


def main():
    output.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.evaluation_mode = True
    cfg.tactile.sides = ("left", "right")
    cfg.tactile.calibration_mode = args.tactile_calibration
    cfg.task.contact_force["log_contact_force"] = False
    object_path = Path(cfg.task.held_asset_cfg.usd_path)
    metadata = {
        "schema": "tactile_pose_v2", "dataset_uuid": str(uuid.uuid4()), "status": "collecting",
        "task": args.task, "root_seed": args.seed, "num_envs": args.num_envs,
        "requested_grasps": args.grasps, "frames_per_grasp": args.frames_per_grasp,
        "requested_rotation_range_deg": list(args.rotation_range_deg),
        "contact_depth_threshold_mm": args.contact_depth_mm, "contact_min_pixels": args.contact_min_pixels,
        "gripper_frame": "panda_hand rigid-body link frame", "object_frame": "HeldAsset root rigid-body link frame",
        "rotation_convention": "R_GO = transpose(R_WG) @ R_WO; active rotations, column vectors; quaternion wxyz",
        "position_convention": "p_GO = transpose(R_WG) @ (p_WO - p_WG); world positions include env origins",
        "opening_convention": "Actual panda_finger_joint1 + panda_finger_joint2 displacement, meters; not calibrated gel surface clearance",
        "contact_convention": "Geometric contact proxy: each side has at least contact_min_pixels above contact_depth_threshold_mm",
        "status_bits": {"1": "left_contact_invalid", "2": "right_contact_invalid", "4": "nonfinite", "8": "pose_or_opening_invalid"},
        "objects": {"0": {"task": args.task, "asset": str(object_path), "sha256": hashlib.sha256(object_path.read_bytes()).hexdigest()}},
        "mechanics": "PhysX rigid gelpads", "optics": "Taxim", "sensor": "GelSight Mini",
        "collection_mode": "Independent randomized resets; zero-action hold after physical finger closure; no policy checkpoint",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = None
    writer = None
    try:
        env = PoseCollectionEnv(cfg, render_mode="rgb_array")
        if args.frames_per_grasp >= env.max_episode_length - 1:
            raise ValueError("frames_per_grasp must end before the task's automatic episode reset")
        metadata.update({"physics_dt_s": env.physics_dt, "control_dt_s": env.step_dt,
                         "robot_joint_names": env._robot.joint_names,
                         "tactile_resolution_wh": list(cfg.tactile.resolution),
                         "tactile_calibration": env.tactile_calibration,
                         "camera_depth_convention": "Image-plane depth in mm; nonfinite rays replaced by far clip"})
        writer = PoseWriter(output / "samples.h5", env, metadata)
        with torch.inference_mode():
            for start in range(0, args.grasps, args.num_envs):
                ids = np.arange(start, start + args.num_envs)
                seeds = args.seed + ids
                angles = np.stack([np.random.default_rng(int(seed)).uniform(-np.array(args.rotation_range_deg),
                                  np.array(args.rotation_range_deg)) for seed in seeds])
                quat = Rotation.from_euler("xyz", angles, degrees=True).as_quat()[:, [3, 0, 1, 2]]
                env.recorder = None
                env.grasp_perturbation_wxyz = torch.as_tensor(quat, device=env.device, dtype=torch.float32)
                reset_seed = int(args.seed + start)
                env.reset(seed=reset_seed)
                env.refresh_tactile()
                writer.begin_grasps(ids, seeds, reset_seed, angles)
                env.recorder = writer
                for _ in range(args.frames_per_grasp):
                    if not app.is_running():
                        raise RuntimeError("Simulation stopped before collection completed")
                    env.step(torch.zeros_like(env.actions))
                env.recorder = None
                writer.flush()
                print(f"POSE_COLLECT grasps={start + args.num_envs}/{args.grasps} "
                      f"samples={writer.count} valid={writer.valid_count}", flush=True)
        writer.finish()
        metadata.update(status="complete", samples=writer.count, valid_samples=writer.valid_count,
                        completed_grasps=args.grasps)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("POSE_DATA_REPORT " + json.dumps(metadata), flush=True)
        print(f"OUTPUT {output}", flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=str(error))
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        raise
    finally:
        if env is not None:
            env.recorder = None
        try:
            if writer is not None:
                writer.close()
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
