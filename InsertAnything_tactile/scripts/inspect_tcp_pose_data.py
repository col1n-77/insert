"""Validate v3 TCP labels/contact depth and export actual stereo RGB previews."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tactile_pose.tcp_sampling import SCHEMA


def inspect(path, output):
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema") != SCHEMA or f.attrs.get("status") != "complete":
            raise ValueError(f"Not a completed v3 TCP dataset: {path}")
        meta = json.loads(f.attrs["metadata_json"])
        valid = f["quality/sample_valid"][:]
        n = len(valid)
        assert n > 0 and valid.any()
        assert np.array_equal(valid, f["quality/status_code"][:] == 0)
        # h5py requires increasing column indices; convert to NumPy first.
        rt = Rotation.from_quat(f["tcp/quaternion_wxyz"][:][:, [1, 2, 3, 0]])
        ro = Rotation.from_quat(f["object/quaternion_wxyz"][:][:, [1, 2, 3, 0]])
        expected_r = rt.as_matrix().transpose(0, 2, 1) @ ro.as_matrix()
        expected_p = np.einsum("nij,nj->ni", rt.as_matrix().transpose(0, 2, 1),
                                f["object/position_w_m"][:] - f["tcp/position_w_m"][:])
        error_p = float(np.max(np.abs(f["target/position_to_m"][:] - expected_p)))
        error_r = float(np.max(np.abs(f["target/rotation_to"][:] - expected_r)))
        assert error_p < 1e-6 and error_r < 1e-6, (error_p, error_r)
        target_r = f["target/rotation_to"][:]
        target_q = Rotation.from_quat(f["target/quaternion_to_wxyz"][:][:, [1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(target_r, target_q, atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(target_r), 1, atol=1e-6)
        approach = np.einsum("nij,nj->ni", target_r, f["sampling/approach_direction_o"][:])
        np.testing.assert_allclose(approach, np.tile([0, 0, 1], (n, 1)), atol=1e-5)
        center = np.einsum("nij,j->ni", target_r, meta["object_bbox_center_o_m"]) + f["target/position_to_m"][:]
        np.testing.assert_allclose(center, f["sampling/requested_center_t_m"][:], atol=1e-6)
        assert np.unique(f["ids/contact_pose_id"][:]).size == n
        assert np.unique(f["random/seed"][:]).size == n
        assert np.unique(f["time/sim_time_s"][:]).size == 1
        assert np.unique(f["time/physics_step"][:]).size == 1
        assert np.all(np.diff(f["time/unix_ns"][:]) > 0)
        assert np.all(np.diff(f["time/render_refresh_index"][:]) > 0)
        actual_q = f["gripper/finger_joint_position_m"][:]
        limits = np.array(meta["finger_joint_limits_m"])
        assert (actual_q >= limits[:, 0] - 1e-6).all() and (actual_q <= limits[:, 1] + 1e-6).all()
        np.testing.assert_allclose(actual_q.sum(1), f["gripper/opening_joint_sum_m"][:, 0], atol=1e-7)
        proposal_count = sum(1 for _ in path.with_name("proposals.jsonl").open())
        assert proposal_count == meta["proposals"]
        direction_ids = f["ids/direction_id"][:]
        per_direction = np.bincount(direction_ids[valid], minlength=len(meta["approach_directions_o"]))
        assert np.all(per_direction == meta["samples_per_direction"])
        report = {"task": meta["task"], "path": str(path), "status": "passed", "rendered_attempts": n,
            "valid_samples": int(valid.sum()), "proposals": proposal_count,
            "accepted_per_direction": per_direction.tolist(), "max_position_label_error_m": error_p,
            "max_rotation_matrix_label_error": error_r, "sides": {}}
        for side in ("left", "right"):
            peak = f[f"contact/{side}_max_press_mm"][:]
            np.testing.assert_allclose(peak[valid], f["sampling/requested_depth_mm"][:][valid],
                                       atol=meta["depth_tolerance_mm"], rtol=0)
            raw_max_error = 0.
            hashes = set()
            for row in range(n):
                press = f[f"tactile/{side}/press_depth_mm"][row]
                depth = f[f"tactile/{side}/camera_depth_mm"][row]
                zero = meta["tactile_calibration"]["zero_reference_mm"][side]
                expected = np.clip(zero - depth, 0, meta["tactile_calibration"]["gel_height_mm"])
                raw_max_error = max(raw_max_error, float(np.max(np.abs(press - expected))))
                assert np.isfinite(press).all() and np.isfinite(depth).all()
                assert float(press.max()) == float(peak[row])
                if valid[row]:
                    image = f[f"tactile/{side}/rgb"][row]
                    assert image.shape == (240, 320, 3) and image.dtype == np.uint8
                    assert (press > meta["contact_depth_threshold_mm"]).sum() >= meta["contact_min_pixels"]
                    hashes.add(hashlib.sha256(image.tobytes()).hexdigest())
            assert raw_max_error < 1e-4
            report["sides"][side] = {"valid_depth_range_mm": [float(peak[valid].min()), float(peak[valid].max())],
                "max_valid_depth_error_mm": float(np.max(np.abs(peak[valid]-f["sampling/requested_depth_mm"][:][valid]))),
                "raw_depth_to_press_max_error_mm": raw_max_error, "unique_valid_rgb_images": len(hashes)}
        selected = [int(np.flatnonzero(valid & (direction_ids == i))[0]) for i in range(len(per_direction))]
        fig, axes = plt.subplots(len(selected), 2, figsize=(8, 3 * len(selected)), squeeze=False, layout="constrained")
        for i, row in enumerate(selected):
            for j, side in enumerate(("left", "right")):
                axes[i, j].imshow(f[f"tactile/{side}/rgb"][row])
                peak = float(f[f"contact/{side}_max_press_mm"][row])
                roll = float(f["sampling/roll_about_approach_deg"][row])
                axes[i, j].set_title(f"Direction {i} | {side} | depth {peak:.3f} mm\nroll {roll:.1f} deg | row {row}", fontsize=10)
                axes[i, j].axis("off")
        fig.suptitle(f"{path.parent.name} | independent TCP contact poses", fontsize=14)
        fig.savefig(output / f"{path.parent.name}_directions.png", dpi=140)
        plt.close(fig)
        # Representative = valid depth closest to that object's median.
        eligible = np.flatnonzero(valid)
        depths = f["sampling/requested_depth_mm"][:][eligible]
        representative = int(eligible[np.argmin(np.abs(depths-np.median(depths)))])
        preview = {s: f[f"tactile/{s}/rgb"][representative] for s in ("left", "right")}
        preview.update(row=representative, depth=float(f["sampling/requested_depth_mm"][representative]))
        return report, preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="collection directory, task directory or samples.h5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error(f"Refusing overwrite: {args.output}")
    if args.input.is_file(): paths = [args.input.resolve()]
    elif (args.input / "collection.json").is_file():
        index = json.loads((args.input / "collection.json").read_text())
        if index["status"] != "complete": raise ValueError("Collection is incomplete")
        paths = [Path(d["path"]) for d in index["datasets"]]
    else: paths = [args.input / "samples.h5"]
    args.output.mkdir(parents=True)
    reports, previews = [], []
    for path in paths:
        report, preview = inspect(path, args.output)
        reports.append(report); previews.append(preview)
    fig, axes = plt.subplots(len(paths), 2, figsize=(8, 3 * len(paths)), squeeze=False, layout="constrained")
    for i, (path, preview) in enumerate(zip(paths, previews)):
        for j, side in enumerate(("left", "right")):
            axes[i, j].imshow(preview[side])
            axes[i, j].set_title(f"{path.parent.name} | {side}\nrow {preview['row']} | requested {preview['depth']:.3f} mm", fontsize=10)
            axes[i, j].axis("off")
    fig.suptitle("TCP pose collection v3 | native stereo tactile RGB", fontsize=14)
    fig.savefig(args.output / "stereo_overview.png", dpi=140)
    plt.close(fig)
    result = {"status": "passed", "datasets": reports, "total_valid_samples": sum(r["valid_samples"] for r in reports)}
    (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
