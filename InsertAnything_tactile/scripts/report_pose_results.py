"""Plot held-out angle predictions, representative tactile pairs, and grasp-level uncertainty."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation
    report = json.loads((root / "metrics.json").read_text())
    arrays = np.load(root / "predictions.npz")
    with (root / "predictions.csv").open() as stream:
        records = list(csv.DictReader(stream))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    reference = ckpt["baseline_rotation"].numpy()
    target, predicted, errors = arrays["target_rotation_go"], arrays["rotation_go"], arrays["error_deg"]
    gt = Rotation.from_matrix(reference.T @ target).as_rotvec(degrees=True)
    pred = Rotation.from_matrix(reference.T @ predicted).as_rotvec(degrees=True)
    baseline_error = Rotation.from_matrix(reference.T @ target).magnitude() * 180 / np.pi
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(record["grasp_group"], []).append(index)
    # One representative per grasp in the scatter, so correlated frames do
    # not visually inflate the effective number of independent examples.
    representative = np.array([rows[len(rows)//2] for rows in groups.values()])
    means = np.array([np.mean(errors[rows]) for rows in groups.values()])
    rng = np.random.default_rng(42)
    bootstrap = rng.choice(means, size=(5000, len(means)), replace=True).mean(1)
    ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), layout="constrained")
    for index in range(3):
        ax = axes[index]
        x, y = gt[representative, index], pred[representative, index]
        lo, hi = min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2
        ax.plot([lo, hi], [lo, hi], "--", color="0.5", linewidth=1)
        ax.scatter(x, y, c=errors[representative], cmap="viridis_r", vmin=0, vmax=15, s=15, alpha=0.8)
        ax.set(xlabel="True angle component (deg)", ylabel="Predicted (deg)",
               title=f"{'XYZ'[index]} | relative to training mean", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_aspect("equal", adjustable="box")
    for values, label, color in ((errors, "Stereo CNN + opening", "tab:blue"),
                                  (baseline_error, "Training-mean baseline", "0.5")):
        axes[3].plot(np.sort(values), np.arange(1, len(values)+1) / len(values), label=label, color=color)
    axes[3].axvline(5, linestyle=":", color="tab:green", linewidth=1)
    axes[3].set(xlabel="SO(3) rotation error (deg)", ylabel="Fraction of valid frames", title="Error distribution", ylim=(0, 1), xlim=(0, max(40, np.percentile(errors, 99))))
    axes[3].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{report['split']}: {len(groups)} independent grasps | mean {errors.mean():.2f} deg | P95 {np.percentile(errors,95):.2f} deg")
    fig.savefig(root / "angle_predictions.png", dpi=170)
    plt.close(fig)

    # Fixed quantiles of grasp-level error, including the worst; do not select
    # only attractive examples. Use the middle valid frame of each grasp.
    order = np.argsort(means)
    chosen = order[np.rint(np.linspace(0, len(order)-1, min(6, len(order)))).astype(int)]
    fig, axes = plt.subplots(2, len(chosen), figsize=(2.65 * len(chosen), 5.1), layout="constrained", squeeze=False)
    examples = []
    for col, group_index in enumerate(chosen):
        index = representative[group_index]
        record = records[index]
        row = int(record["source_row"])
        with h5py.File(record["source_h5"], "r") as f:
            grasp = int(f["ids/grasp_id"][row])
            for side_index, side in enumerate(("left", "right")):
                axes[side_index, col].imshow(f[f"tactile/{side}/rgb"][row])
                axes[side_index, col].axis("off")
                axes[side_index, col].set_title(f"{side} | grasp {grasp}" if side_index == 0 else f"frame error {errors[index]:.2f} deg")
        examples.append({"grasp_group": record["grasp_group"], "source_row": row, "error_deg": float(errors[index]),
                         "truth_rotvec_about_training_mean_deg": gt[index].tolist(),
                         "prediction_rotvec_about_training_mean_deg": pred[index].tolist()})
    fig.suptitle("Tactile pairs across the grasp-error distribution (best to worst)")
    fig.savefig(root / "tactile_prediction_examples.png", dpi=155)
    plt.close(fig)
    detailed = {"split": report["split"], "independent_grasps": len(groups),
                "mean_grasp_error_deg": float(means.mean()), "mean_grasp_error_bootstrap_95ci_deg": ci,
                "scope": "Bootstrap resamples independent grasps, not correlated frames; no real-world extrapolation",
                "plot_angles": "Rotation-vector components relative to the fixed training-only mean rotation; not Euler angles",
                "selected_examples": examples}
    (root / "analysis.json").write_text(json.dumps(detailed, indent=2) + "\n")
    print(json.dumps({key: value for key, value in detailed.items() if key != "selected_examples"}, indent=2))


if __name__ == "__main__":
    main()
