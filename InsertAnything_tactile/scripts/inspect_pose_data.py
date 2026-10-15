"""Verify pose labels against SciPy and plot actual, valid rotation coverage."""
import argparse
import json
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    output = args.data.resolve().parent
    with h5py.File(args.data, "r") as f:
        assert f.attrs["schema"] in ("tactile_pose_v1", "tactile_pose_v2") and f.attrs["status"] == "complete"
        metadata = json.loads(f.attrs["metadata_json"])
        n = len(f["ids/grasp_id"])
        datasets = []
        def check(name, item):
            if isinstance(item, h5py.Dataset):
                assert len(item) == n, name
                for start in range(0, n, 16):
                    assert np.isfinite(item[start:start+16]).all(), name
                datasets.append(name)
        f.visititems(check)
        rg = Rotation.from_quat(f["gripper/quaternion_wxyz"][:][:, [1, 2, 3, 0]])
        ro = Rotation.from_quat(f["object/quaternion_wxyz"][:][:, [1, 2, 3, 0]])
        expected = (rg.inv() * ro).as_matrix()
        actual = f["target/rotation_go"][:]
        error = float(np.abs(expected - actual).max())
        np.testing.assert_allclose(expected, actual, atol=2e-6)
        p_go = rg.inv().apply(f["object/position_w_m"][:] - f["gripper/position_w_m"][:])
        np.testing.assert_allclose(p_go, f["target/position_go_m"][:], atol=2e-6)
        qlabel = Rotation.from_quat(f["target/quaternion_go_wxyz"][:][:, [1, 2, 3, 0]])
        np.testing.assert_allclose(qlabel.as_matrix(), actual, atol=2e-6)
        np.testing.assert_allclose(actual.transpose(0, 2, 1) @ actual, np.tile(np.eye(3), (n, 1, 1)), atol=2e-6)
        np.testing.assert_allclose(np.linalg.det(actual), 1, atol=2e-6)
        np.testing.assert_allclose(f["gripper/finger_joint_position_m"][:].sum(-1),
                                   f["gripper/opening_joint_sum_m"][:, 0], atol=1e-8)
        for side in ("left", "right"):
            assert f[f"tactile/{side}/rgb"].shape == (n, 240, 320, 3)
            assert f[f"tactile/{side}/rgb"].dtype == np.uint8
            for start in range(0, n, 16):
                press = f[f"tactile/{side}/press_depth_mm"][start:start+16]
                if f.attrs["schema"] == "tactile_pose_v2":
                    calibration = metadata["tactile_calibration"]
                    depth = f[f"tactile/{side}/camera_depth_mm"][start:start+16]
                    reference = calibration["zero_reference_mm"][side]
                    expected_press = np.clip(reference - depth, 0, calibration["gel_height_mm"])
                    near, far = np.asarray(calibration["clipping_range_m"]) * 1000
                    expected_press[(depth < near) | (depth > far)] = 0
                    np.testing.assert_allclose(press, expected_press, atol=4e-6)
                pixels = (press > metadata["contact_depth_threshold_mm"]).sum((1, 2))
                np.testing.assert_array_equal(pixels, f[f"contact/{side}_pixels"][start:start+16])
                np.testing.assert_array_equal(pixels >= metadata["contact_min_pixels"],
                                              f[f"contact/{side}_valid"][start:start+16])
        valid = f["quality/sample_valid"][:]
        np.testing.assert_array_equal(valid, f["quality/status_code"][:] == 0)
        assert np.all(f["contact/both_valid"][:][valid])
        groups = f["ids/grasp_id"][:]
        steps = f["time/grasp_step"][:]
        times = f["time/grasp_time_s"][:]
        np.testing.assert_allclose(times, steps * metadata["control_dt_s"])
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            assert len(np.unique(f["ids/env_id"][:][indices])) == 1
            assert len(np.unique(f["random/seed"][:][indices])) == 1
            np.testing.assert_array_equal(steps[indices], np.arange(1, len(indices)+1))
            assert np.all(np.diff(f["time/sim_time_s"][:][indices]) > 0)
        r = Rotation.from_matrix(actual[valid])
        relative = (r.mean().inv() * r).as_rotvec(degrees=True)
        ids, counts = np.unique(f["quality/status_code"][:], return_counts=True)
        width = f["gripper/opening_joint_sum_m"][:][valid, 0] * 1000
        report = {
            "status": "passed", "samples": n, "valid_samples": int(valid.sum()),
            "grasps": len(np.unique(groups)), "valid_grasps": len(np.unique(groups[valid])),
            "dataset_fields": len(datasets), "label_max_matrix_error_vs_scipy": error,
            "tactile_calibration": metadata.get("tactile_calibration", {"id": "legacy_shared_28mm"}),
            "status_counts": {str(i): int(c) for i, c in zip(ids, counts)},
            "actual_rotvec_deviation_from_mean_deg": {"min": relative.min(0).tolist(), "max": relative.max(0).tolist()},
            "rotation_covariance_eigenvalues_deg2": np.linalg.eigvalsh(np.cov(relative.T)).tolist(),
            "actual_joint_sum_range_mm": [float(width.min()), float(width.max())],
            "notes": "Coverage is the physically realized relative rotation, not requested reset angles. Contact validity is geometric."
        }
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
        for axis, ax in enumerate(axes):
            ax.hist(relative[:, axis], bins=20, color=("tab:blue", "tab:orange", "tab:green")[axis])
            ax.set_xlabel(f"{'XYZ'[axis]} rotation-vector deviation from mean (deg)")
            ax.set_ylabel("Valid samples")
        fig.suptitle("Actual object-to-gripper rotation coverage (valid contacts)")
        fig.savefig(output / "rotation_coverage.png", dpi=150)
        plt.close(fig)
        choices = []
        for group in np.unique(groups[valid]):
            choices.append(np.flatnonzero((groups == group) & valid)[0])
        choices = choices[::max(1, len(choices)//6)][:6]
        fig, axes = plt.subplots(2, len(choices), figsize=(2.4 * len(choices), 4.8), squeeze=False, layout="constrained")
        for col, row in enumerate(choices):
            for index, side in enumerate(("left", "right")):
                axes[index, col].imshow(f[f"tactile/{side}/rgb"][row])
                axes[index, col].axis("off")
                axes[index, col].set_title(f"Grasp {groups[row]} | {side}")
        fig.savefig(output / "grasp_preview.png", dpi=130)
        plt.close(fig)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
