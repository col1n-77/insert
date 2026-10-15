"""One Isaac Sim process per task; use collect_tcp_pose_data.py as the entrypoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--task", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
from PIL import Image
import insertanything_tactile
from insertanything_tactile.env import InsertAnythingTactileEnv
from isaaclab_tasks.utils import parse_env_cfg
from tactile_pose.tcp_sampling import TASK_SUFFIXES, SCHEMA, STATUS_BITS, propose_pose
from tactile_pose.tcp_sim_collection import StaticContactSampler, ContactWriter


def main():
    spec = json.loads(args.config.read_text())
    output = Path(spec["output"]) / args.task
    output.mkdir(parents=True, exist_ok=False)
    task_id = f"InsertAnythingTactile-{args.task}-Direct-v0"
    object_id = TASK_SUFFIXES.index(args.task)
    cfg = parse_env_cfg(task_id, device=args.device or "cuda:0", num_envs=1)
    cfg.seed = spec["seed"] + object_id
    cfg.evaluation_mode = True
    cfg.tactile.calibration_mode = "geometry_per_side"
    asset = Path(cfg.task.held_asset_cfg.usd_path)
    directions = spec["approach_directions"]
    quota = spec["samples_per_task"] // len(directions)
    meta = {
        "schema": SCHEMA, "dataset_uuid": str(uuid.uuid4()), "status": "initializing", "task": task_id,
        "object_id": object_id, "object_asset": str(asset), "object_asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        "root_seed": spec["seed"], "initialization_seed": cfg.seed, "num_envs": 1,
        "requested_valid_samples": spec["samples_per_task"], "samples_per_direction": quota,
        "approach_directions_o": directions, "roll_range_deg": spec["roll_range_deg"],
        "offset_xy_range_mm": spec["offset_xy_mm"], "center_z_mm": spec["center_z_mm"],
        "depth_range_mm": spec["depth_range_mm"], "depth_tolerance_mm": spec["depth_tolerance_mm"],
        "max_depth_iterations": spec["max_depth_iterations"], "contact_min_pixels": spec["contact_min_pixels"],
        "contact_depth_threshold_mm": spec["contact_depth_mm"], "status_bits": STATUS_BITS,
        "tcp_frame": "actual panda_fingertip_centered rigid-body link frame",
        "object_frame": "HeldAsset root link; its bounding-box center is placed near TCP",
        "rotation_convention": "R_TO = transpose(R_WT) @ R_WO; active rotations, column vectors, quaternions wxyz",
        "position_convention": "p_TO = transpose(R_WT) @ (p_WO - p_WT); object root origin in TCP meters",
        "approach_convention": "TCP +Z expressed in object coordinates; finger closure along TCP +/-Y",
        "roll_convention": "Rotation of gripper frame about approach; object-to-TCP rotation is its inverse",
        "sampling_distribution": "Uniform proposals; acceptance conditioned on joint limits, dual contact and depth tolerance. Accepted depth/offset/roll are NOT necessarily uniform.",
        "ids_convention": "dataset_uuid + contact_pose_id uniquely identifies an independent proposal; grasp_id aliases contact_pose_id for grouping, no physical grasp episode",
        "opening_convention": "joint sum is actual joint displacement; gel_front_gap_m additionally includes the measured mount offset",
        "collection_mode": "static_geometric_contact_rendering; pose writes + sim.forward + render; NO physics steps between samples",
        "time_convention": "unix_ns is wall clock; sim_time/physics_step remain fixed after initialization; render_refresh_index counts two-render refreshes",
        "mechanics": "Rigid geometries with prescribed overlap; no dynamic grasp, force estimate, gel FEM or slip model",
        "optics": "Taxim; calibrated per side; native camera orientation; 320x240 uint8 RGB",
        "old_model_compatibility": "v1/v2 panda_hand target/checkpoints are not the TCP target; v3 has a distinct schema and needs a compatible training data adapter",
    }
    env = writer = None
    proposal_count = skipped_bounds = 0
    accepted_per_direction = [0] * len(directions)
    try:
        env = InsertAnythingTactileEnv(cfg)
        with torch.inference_mode():
            env.reset(seed=cfg.seed)
            sampler = StaticContactSampler(env, meta)
            meta["status"] = "collecting"
            (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
            writer = ContactWriter(output / "samples.h5", meta)
            with (output / "proposals.jsonl").open("x") as journal:
                for direction_id, direction in enumerate(directions):
                    for slot in range(quota):
                        accepted = False
                        for retry in range(spec["max_attempts_per_sample"]):
                            if not app.is_running(): raise RuntimeError("Simulator stopped")
                            proposal = propose_pose(spec["seed"], object_id, direction_id, slot, retry, direction,
                                sampler.center_o, spec["offset_xy_mm"], spec["roll_range_deg"],
                                spec["depth_range_mm"], spec["center_z_mm"])
                            proposal_id = proposal_count
                            proposal_count += 1
                            record = {"proposal_id": proposal_id, "direction_id": direction_id, "slot": slot,
                                      "retry": retry, "seed": proposal["seed"], "depth_mm": proposal["depth_mm"],
                                      "roll_deg": proposal["roll_deg"], "offset_xy_mm": proposal["offset_xy_mm"].tolist()}
                            if not sampler.bounds_feasible(proposal):
                                skipped_bounds += 1
                                record.update(status="rejected_before_render", reason="full_mesh_cannot_reach_target_within_joint_limits")
                                journal.write(json.dumps(record) + "\n")
                                continue
                            row = sampler.capture(proposal, {
                                "ids/sample_id": np.int64(writer.count), "ids/contact_pose_id": np.int64(proposal_id),
                                "ids/grasp_id": np.int64(proposal_id), "ids/object_id": np.int32(object_id),
                                "ids/env_id": np.int32(0), "ids/direction_id": np.int32(direction_id),
                                "ids/slot_id": np.int32(slot), "ids/retry_id": np.int32(retry)})
                            row["ids/accepted_id"] = np.int64(writer.valid_count if row["quality/sample_valid"] else -1)
                            writer.append(row)
                            record.update(status="accepted" if row["quality/sample_valid"] else "rejected_after_render",
                                          h5_row=writer.count-1, status_code=int(row["quality/status_code"]))
                            journal.write(json.dumps(record) + "\n")
                            if row["quality/sample_valid"]:
                                accepted_per_direction[direction_id] += 1
                                if slot == 0:
                                    for side in ("left", "right"):
                                        Image.fromarray(row[f"tactile/{side}/rgb"]).save(output / f"direction_{direction_id}_{side}.png")
                                accepted = True
                                print(f"TCP_PROGRESS {args.task} valid={writer.valid_count}/{spec['samples_per_task']} rendered={writer.count} proposals={proposal_count}", flush=True)
                                writer.file.flush(); journal.flush()
                                break
                        if not accepted:
                            raise RuntimeError(f"Direction {direction_id}, slot {slot}: no valid sample within {spec['max_attempts_per_sample']} proposals. Inspect depth range, offsets and joint limits.")
            meta.update(proposals=proposal_count, proposals_rejected_before_render=skipped_bounds,
                        accepted_per_direction=accepted_per_direction)
            writer.finish("complete")
            print("TCP_REPORT " + json.dumps(meta), flush=True)
    except BaseException as error:
        meta.update(status="failed", error=str(error), proposals=proposal_count,
                    proposals_rejected_before_render=skipped_bounds, accepted_per_direction=accepted_per_direction)
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
