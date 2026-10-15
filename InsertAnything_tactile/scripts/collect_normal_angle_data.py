"""Collect upright pegs and single grasp-normal in-hand angles for all six tasks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np
from tactile_pose.tcp_sampling import TASK_SUFFIXES

ROOT = Path(__file__).resolve().parents[1]


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("all", *TASK_SUFFIXES), default="all")
    p.add_argument("--samples_per_task", type=int, default=100, help="VALID pairs per task, divisible by angle_bins")
    p.add_argument("--seed", type=int, default=80000)
    p.add_argument("--angle_range_deg", nargs=2, type=float, default=(-15., 15.))
    p.add_argument("--angle_bins", type=int, default=5, help="Equal valid quotas across angle intervals")
    p.add_argument("--object_yaw_deg", type=float, default=180., help="Fixed world yaw, never randomly sampled")
    p.add_argument("--offset_xy_mm", nargs=2, type=float, default=(2., 2.))
    p.add_argument("--center_z_mm", type=float, default=0., help="Object bbox center axial offset from TCP")
    p.add_argument("--depth_range_mm", nargs=2, type=float, default=(.2, 1.))
    p.add_argument("--depth_tolerance_mm", type=float, default=.03)
    p.add_argument("--contact_depth_mm", type=float, default=.02)
    p.add_argument("--contact_min_pixels", type=int, default=64)
    p.add_argument("--max_depth_iterations", type=int, default=12)
    p.add_argument("--max_attempts_per_sample", type=int, default=1000, help="Bounded retries including cheap geometric rejections")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path)
    return p


def main():
    parser = make_parser()
    args = parser.parse_args()
    if (not np.isfinite(args.angle_range_deg).all() or
            not -45 <= args.angle_range_deg[0] < args.angle_range_deg[1] <= 45):
        parser.error("Use a finite ascending angle range within -45 to +45 degrees")
    if args.angle_bins < 1 or args.samples_per_task < 1 or args.samples_per_task % args.angle_bins:
        parser.error("samples_per_task must be positive and divisible by angle_bins")
    edges = np.linspace(*args.angle_range_deg, args.angle_bins + 1)
    numbers = [*args.offset_xy_mm, args.center_z_mm, args.object_yaw_deg, *args.depth_range_mm,
               args.depth_tolerance_mm, args.contact_depth_mm]
    if not np.isfinite(numbers).all() or args.seed < 0 or min(args.offset_xy_mm) < 0:
        parser.error("Use finite ranges and nonnegative seed/offsets")
    if not (0 < args.depth_range_mm[0] <= args.depth_range_mm[1] <= 2.):
        parser.error("Use a positive ordered depth range, at most the asset's 2 mm indentation limit")
    if not (0 < args.contact_depth_mm < args.depth_range_mm[0]) or args.depth_tolerance_mm <= 0:
        parser.error("Contact threshold must be below requested indentation; tolerance must be positive")
    if min(args.max_attempts_per_sample, args.max_depth_iterations, args.contact_min_pixels) < 1:
        parser.error("Counts must be positive")
    output = (args.output or ROOT / "outputs" / datetime.now().strftime("normal_angle_v4_%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists(): parser.error(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    config = vars(args).copy()
    config.update(output=str(output), angle_edges_deg=edges.tolist())
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    tasks = TASK_SUFFIXES if args.task == "all" else (args.task,)
    index = {"schema": "tactile_normal_collection_v4", "collection_uuid": str(uuid.uuid4()),
             "started_utc": datetime.now(timezone.utc).isoformat(), "status": "collecting", "datasets": []}
    index_path = output / "collection.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    try:
        for task in tasks:
            command = [sys.executable, str(ROOT / "scripts/_collect_normal_angle_worker.py"),
                       "--config", str(config_path), "--task", task, "--device", args.device]
            if args.headless: command.append("--headless")
            log_path = output / f"{task}.log"
            print(f"NORMAL_COLLECT starting {task}; log={log_path}", flush=True)
            with log_path.open("x") as log:
                # Separate processes release Kit and RTX resources between assets.
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while True:
                        try:
                            code = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            lines = log_path.read_text(errors="replace").splitlines()
                            progress = [line for line in lines if line.startswith("NORMAL_PROGRESS")]
                            print(progress[-1] if progress else f"NORMAL_COLLECT initializing {task}", flush=True)
                except BaseException:
                    child.terminate()
                    try: child.wait(timeout=15)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    raise
            if code:
                raise RuntimeError(f"{task} exited {code}; see {log_path}")
            metadata = json.loads((output / task / "metadata.json").read_text())
            index["datasets"].append({"task": task, "path": str(output / task / "samples.h5"),
                                      "dataset_uuid": metadata["dataset_uuid"], "status": metadata["status"],
                                      "valid_samples": metadata["valid_samples"], "rendered_attempts": metadata["attempts"],
                                      "proposals": metadata["proposals"]})
            index_path.write_text(json.dumps(index, indent=2) + "\n")
            print(f"NORMAL_COLLECT finished {task}: valid={metadata['valid_samples']} rendered={metadata['attempts']} proposals={metadata['proposals']}", flush=True)
        index.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        index.update(status="failed", error=str(error))
        raise
    finally:
        index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"OUTPUT {output}", flush=True)


if __name__ == "__main__": main()
