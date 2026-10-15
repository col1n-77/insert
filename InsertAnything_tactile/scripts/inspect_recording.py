"""Offline validation and plots for the completed evaluation recording."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("recording", type=Path, help="Completed record_eval.py output directory")
args = parser.parse_args()
root = args.recording.resolve()
report = json.loads((root / "report.json").read_text())
assert report["status"] == "complete", report["status"]
n = report["frames"]
checks = {"frames": n, "datasets": {}, "videos": {}, "tactile": {}}
with h5py.File(root / "trajectory.h5", "r") as f:
    assert f.attrs["status"] == "complete"
    def check_dataset(name, item):
        if not isinstance(item, h5py.Dataset):
            return
        assert item.shape[0] == n, (name, item.shape)
        for start in range(0, n, 16):
            assert np.isfinite(item[start:start + 16]).all(), name
        checks["datasets"][name] = {"shape": list(item.shape), "dtype": str(item.dtype)}
    f.visititems(check_dataset)
    assert np.array_equal(f["step"][:], np.arange(1, n + 1))
    t = f["time_s"][:]
    np.testing.assert_allclose(t, np.arange(1, n + 1) * report["control_dt_s"])
    done = f["terminated"][:] | f["truncated"][:]
    assert np.flatnonzero(done).tolist() == [n - 1]
    succeeded = f["metrics/episode_succeeded"][:]
    assert int(succeeded[-1]) == report["successful_trials"]
    index = int(f["metrics/first_success_step"][-1]) - 1 if succeeded[-1] else n // 2
    frames = {}
    for name in ("scene", "left_tactile", "right_tactile"):
        cap = cv2.VideoCapture(str(root / f"{name}.mp4"))
        assert cap.isOpened(), name
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert abs(fps - report["sampling_hz"]) < 1e-3, (name, fps)
        count = 0
        shape = None
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            shape = bgr.shape
            assert np.any(bgr), (name, count, "empty")
            if count == index:
                frames[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            count += 1
        cap.release()
        assert count == n, (name, count, n)
        checks["videos"][name] = {"decoded_frames": count, "fps": fps, "shape": list(shape)}

    depth_traces = {}
    for side in ("left", "right"):
        rgb = f[f"tactile/{side}/rgb"]
        depth = f[f"tactile/{side}/press_depth_mm"]
        camera = f[f"tactile/{side}/camera_depth_mm"]
        assert rgb.shape == (n, 240, 320, 3) and rgb.dtype == np.uint8
        assert depth.shape == camera.shape == (n, 240, 320)
        assert depth.dtype == camera.dtype == np.float32
        baseline = rgb[0].astype(np.float32)
        changes = []
        peaks = []
        for start in range(0, n, 16):
            batch = depth[start:start + 16]
            assert batch.min() >= 0
            peaks.extend(batch.max(axis=(1, 2)).tolist())
            changes.extend(np.abs(rgb[start:start + 16].astype(np.float32) - baseline).mean(axis=(1, 2, 3)).tolist())
        assert max(changes) > 0.01, (side, "static tactile sequence")
        depth_traces[side] = np.asarray(peaks)
        checks["tactile"][side] = {
            "max_press_depth_mm": max(peaks),
            "max_rgb_mean_abs_change_from_first": max(changes),
            "video_raw_mean_abs_error_at_preview": float(np.abs(frames[f"{side}_tactile"].astype(np.float32) - rgb[index]).mean()),
        }
        assert checks["tactile"][side]["video_raw_mean_abs_error_at_preview"] < 10

    with (root / "states.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == n
    np.testing.assert_allclose([float(row["time_s"]) for row in rows], t)
    force = f["force/contact_local_n"][:]
    for axis in range(3):
        np.testing.assert_allclose([float(row[f"force_contact_local_n_{axis}"]) for row in rows], force[:, axis])

    fig = plt.figure(figsize=(13, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(frames["scene"])
    ax.set_title(f"Policy evaluation | t = {t[index]:.2f} s")
    ax.axis("off")
    for row, side in enumerate(("left", "right")):
        ax = fig.add_subplot(grid[row, 1])
        ax.imshow(f[f"tactile/{side}/rgb"][index])
        ax.set_title(f"{side.title()} GelSight Mini RGB")
        ax.axis("off")
        ax = fig.add_subplot(grid[row, 2])
        im = ax.imshow(f[f"tactile/{side}/press_depth_mm"][index], cmap="magma", vmin=0,
                       vmax=max(checks["tactile"][side]["max_press_depth_mm"], 0.1))
        ax.set_title(f"{side.title()} geometric press depth")
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="mm", shrink=0.7)
    fig.savefig(root / "preview.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for axis, label in enumerate(("Fx", "Fy", "Fz")):
        axes[0].plot(t, force[:, axis], label=label, linewidth=1)
    axes[0].plot(t, np.linalg.norm(force, axis=1), label="norm", linewidth=1.2, color="black")
    axes[0].set_ylabel("Processed contact force (N)")
    axes[0].legend(ncol=4)
    ee = f["robot/ee_position_env_m"][:]
    for axis, label in enumerate(("X", "Y", "Z")):
        axes[1].plot(t, (ee[:, axis] - ee[0, axis]) * 1000, label=label)
    axes[1].set_ylabel("EE displacement from first (mm)")
    axes[1].legend(ncol=3)
    for side in ("left", "right"):
        axes[2].plot(t, depth_traces[side], label=side)
    axes[2].set_ylabel("Maximum geometric press (mm)")
    axes[2].set_xlabel("Policy rollout time (s)")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
        if report["first_success_time_s"] is not None:
            ax.axvline(report["first_success_time_s"], linestyle="--", color="green", alpha=0.6)
    fig.suptitle("Recorded signals | dashed line: first insertion success")
    fig.savefig(root / "signals.png", dpi=150)
    plt.close(fig)

checks["status"] = "passed"
checks["csv_rows"] = len(rows)
checks["duration_s"] = float(t[-1])
checks["all_datasets_finite"] = True
checks["time_steps_contiguous"] = True
checks["only_last_sample_is_terminal"] = True
(root / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps({key: value for key, value in checks.items() if key != "datasets"}, indent=2))
print("Datasets validated:", len(checks["datasets"]))
