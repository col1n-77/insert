"""One-task worker for upright objects and grasp-normal wrist tilt."""
import argparse, hashlib, json, os, sys, traceback, uuid
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--task", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
from PIL import Image
import insertanything_tactile
from insertanything_tactile.env import InsertAnythingTactileEnv
from isaaclab_tasks.utils import parse_env_cfg
from tactile_pose.tcp_sampling import TASK_SUFFIXES, STATUS_BITS
from tactile_pose.normal_sampling import SCHEMA, CONSTRAINT_BIT, propose_normal_pose
from tactile_pose.normal_sim_collection import NormalContactSampler, NormalWriter


def main():
    spec = json.loads(args.config.read_text())
    output = Path(spec["output"]) / args.task
    output.mkdir(parents=True, exist_ok=False)
    task = f"InsertAnythingTactile-{args.task}-Direct-v0"
    object_id = TASK_SUFFIXES.index(args.task)
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=1)
    cfg.seed = spec["seed"] + object_id; cfg.evaluation_mode = True
    cfg.tactile.calibration_mode = "geometry_per_side"
    asset = Path(cfg.task.held_asset_cfg.usd_path)
    edges = spec["angle_edges_deg"]
    quota = spec["samples_per_task"] // spec["angle_bins"]
    meta = {
        "schema": SCHEMA, "dataset_uuid": str(uuid.uuid4()), "status": "initializing", "task": task,
        "object_id": object_id, "object_asset": str(asset), "object_asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        "root_seed": spec["seed"], "initialization_seed": cfg.seed, "num_envs": 1,
        "requested_valid_samples": spec["samples_per_task"], "samples_per_angle_bin": quota,
        "angle_range_deg": spec["angle_range_deg"], "angle_edges_deg": edges,
        "object_yaw_deg": spec["object_yaw_deg"],
        "offset_xy_range_mm": spec["offset_xy_mm"], "center_z_mm": spec["center_z_mm"],
        "depth_range_mm": spec["depth_range_mm"], "depth_tolerance_mm": spec["depth_tolerance_mm"],
        "max_depth_iterations": spec["max_depth_iterations"], "contact_min_pixels": spec["contact_min_pixels"],
        "contact_depth_threshold_mm": spec["contact_depth_mm"],
        "status_bits": {**STATUS_BITS, CONSTRAINT_BIT: "upright_or_single_axis_constraint_invalid"},
        "tcp_frame": "actual panda_fingertip_centered link",
        "object_frame": "HeldAsset root; local +Z is insertion axis, fixed to world +Z",
        "rotation_convention": "R_TO = R_WT.T @ R_WO; active column vectors, wxyz quaternions",
        "position_convention": "p_TO = R_WT.T @ (p_WO - p_WT), meters",
        "angle_convention": "R_TO(theta) = RotY(theta) @ R_TO(0), R_TO(0)=RotY(180 deg); positive right-hand angle about +TCP Y",
        "pose_constraint": "object +Z world vertical and fixed world yaw; only relative rotation about TCP +Y grasp normal",
        "sampling_distribution": "Uniform proposals within five/default angle bins with equal accepted quotas; offsets/depth are rejection-conditioned",
        "ids_convention": "UUID + contact_pose_id identifies one independent pose; grasp_id is a grouping alias, no physical grasp episode",
        "opening_convention": "finger joint sum is measured displacement, gel_front_gap_m includes installation offset",
        "collection_mode": "static geometry rendering; actual Panda arm joints solved by USD FK/IK; robot base fixed; no physics step between samples",
        "time_convention": "wall-clock unix_ns and render_refresh_index advance; sim_time_s/physics_step are constant after initialization",
        "mechanics": "prescribed rigid overlap; no force, dynamic stability or FEM labels",
        "optics": "Taxim per-side geometry calibration, native 320x240 RGB",
        "prediction_target": "target/in_hand_angle_deg, one scalar; world poses and other rotation components are labels/validation only",
    }
    env = writer = None
    proposals = skipped = 0
    counts = [0] * spec["angle_bins"]
    try:
        env = InsertAnythingTactileEnv(cfg, render_mode="rgb_array")
        with torch.inference_mode():
            env.reset(seed=cfg.seed)
            sampler = NormalContactSampler(env, meta)
            meta["status"] = "collecting"
            (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
            writer = NormalWriter(output / "samples.h5", meta)
            with (output / "proposals.jsonl").open("x") as journal:
                for angle_bin in range(spec["angle_bins"]):
                    for slot in range(quota):
                        accepted = False
                        for retry in range(spec["max_attempts_per_sample"]):
                            if not app.is_running(): raise RuntimeError("Simulator stopped")
                            proposal = propose_normal_pose(spec["seed"], object_id, angle_bin, slot, retry, edges,
                                sampler.center_o, spec["offset_xy_mm"], spec["depth_range_mm"], spec["center_z_mm"])
                            pid = proposals; proposals += 1
                            record = {"proposal_id": pid, "angle_bin": angle_bin, "slot": slot, "retry": retry,
                                      "seed": proposal["seed"], "angle_deg": proposal["angle_deg"],
                                      "depth_mm": proposal["depth_mm"], "offset_xy_mm": proposal["offset_xy_mm"].tolist()}
                            if not sampler.bounds_feasible(proposal):
                                skipped += 1
                                record.update(status="rejected_before_render", reason="full_mesh_joint_limit")
                                journal.write(json.dumps(record) + "\n"); continue
                            row = sampler.capture(proposal, {
                                "ids/sample_id": np.int64(writer.count), "ids/contact_pose_id": np.int64(pid),
                                "ids/grasp_id": np.int64(pid), "ids/object_id": np.int32(object_id),
                                "ids/env_id": np.int32(0), "ids/angle_bin": np.int32(angle_bin),
                                "ids/slot_id": np.int32(slot), "ids/retry_id": np.int32(retry)})
                            row["ids/accepted_id"] = np.int64(writer.valid_count if row["quality/sample_valid"] else -1)
                            writer.append(row)
                            record.update(status="accepted" if row["quality/sample_valid"] else "rejected_after_render",
                                          h5_row=writer.count-1, status_code=int(row["quality/status_code"]))
                            journal.write(json.dumps(record) + "\n")
                            if row["quality/sample_valid"]:
                                counts[angle_bin] += 1
                                if slot == 0:
                                    for side in ("left", "right"):
                                        Image.fromarray(row[f"tactile/{side}/rgb"]).save(output / f"angle_bin_{angle_bin}_{side}.png")
                                    scene = None
                                    # The viewport render product is lazy; its
                                    # first frame may be black until warmed up.
                                    for _ in range(8):
                                        scene = env.render()
                                        if scene is not None and np.any(scene): break
                                        env.sim.render()
                                    if scene is not None:
                                        Image.fromarray(scene[:, :, :3]).save(output / f"angle_bin_{angle_bin}_scene.png")
                                    (output / f"angle_bin_{angle_bin}_preview.json").write_text(json.dumps({
                                        "h5_row": writer.count-1, "angle_deg": float(row["target/in_hand_angle_deg"]),
                                        "object_vertical_error_deg": float(row["quality/object_vertical_error_deg"])}, indent=2))
                                accepted = True
                                print(f"NORMAL_PROGRESS {args.task} valid={writer.valid_count}/{spec['samples_per_task']} rendered={writer.count} proposals={proposals}", flush=True)
                                writer.file.flush(); journal.flush(); break
                        if not accepted:
                            raise RuntimeError(f"No valid pose in angle bin {angle_bin}, slot {slot} after bounded retries")
            meta.update(proposals=proposals, proposals_rejected_before_render=skipped, accepted_per_angle_bin=counts)
            writer.finish("complete")
            print("NORMAL_REPORT " + json.dumps(meta), flush=True)
    except BaseException as error:
        meta.update(status="failed", error=str(error), proposals=proposals,
                    proposals_rejected_before_render=skipped, accepted_per_angle_bin=counts)
        if writer is not None and writer.count: writer.finish("failed")
        raise
    finally:
        (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        if writer is not None: writer.close()
        if env is not None: env.close()


if __name__ == "__main__":
    try: main()
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    app.close()
