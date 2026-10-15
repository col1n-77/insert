"""Evaluate a single-angle model on a fresh collection with disjoint seeds."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tactile_pose.normal_learning import (CHECKPOINT_SCHEMA, NormalAngleNet, AngleDataset,
                                         fresh_manifest, evaluate_angles)
from tactile_pose.tcp_sampling import TASK_SUFFIXES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(); torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA: raise ValueError("Not a one-angle v4 checkpoint")
    manifest = fresh_manifest(args.data, checkpoint["manifest"])
    if args.output.exists(): p.error("Output must be a new directory")
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    dataset = AngleDataset(manifest, "external", checkpoint["config"]["image_size_hw"])
    model = NormalAngleNet(**checkpoint["config"]["model"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    metrics, arrays = evaluate_angles(model, DataLoader(dataset, batch_size=64), args.device, checkpoint["baseline_train_mean_deg"])
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    with (args.output / "predictions.csv").open("x", newline="") as f:
        writer = csv.writer(f); writer.writerow(["file", "h5_row", "object_id", "target_deg", "prediction_deg", "error_deg"])
        for record, truth, prediction in zip(dataset.records, arrays["target_deg"], arrays["prediction_deg"]):
            writer.writerow([manifest["files"][record["file"]], record["row"], record["object_id"], float(truth), float(prediction), float(prediction-truth)])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for obj in np.unique(arrays["object_id"]):
        mask = arrays["object_id"] == obj
        axes[0].scatter(arrays["target_deg"][mask], arrays["prediction_deg"][mask], label=TASK_SUFFIXES[int(obj)], s=22, alpha=.8)
    lo, hi = checkpoint["target_contract"]["angle_range_deg"]
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set(xlabel="Actual in-hand angle (deg)", ylabel="Predicted angle (deg)", title="Fresh seeds | one grasp-normal angle")
    axes[0].legend(fontsize=8)
    axes[1].hist(np.abs(arrays["error_deg"]), bins=20)
    axes[1].set(xlabel="Absolute angle error (deg)", ylabel="Samples", title=f"MAE {metrics['mae_deg']:.3f} deg | P95 {metrics['p95_deg']:.3f} deg")
    fig.savefig(args.output / "angle_predictions.png", dpi=160); plt.close(fig)
    order=np.argsort(np.abs(arrays["error_deg"]))
    selected=order[np.linspace(0,len(order)-1,min(6,len(order))).round().astype(int)]
    fig,axes=plt.subplots(len(selected),2,figsize=(8,3*len(selected)),squeeze=False,layout="constrained")
    for i,index in enumerate(selected):
        example=dataset[int(index)]
        obj=TASK_SUFFIXES[int(arrays["object_id"][index])]
        for j,side in enumerate(("left","right")):
            axes[i,j].imshow(example[side].permute(1,2,0).numpy());axes[i,j].axis("off")
            axes[i,j].set_title(f"{obj} | {side}\ntrue {arrays['target_deg'][index]:+.2f}, pred {arrays['prediction_deg'][index]:+.2f}, error {abs(arrays['error_deg'][index]):.2f} deg",fontsize=10)
    fig.suptitle("Fresh-test examples spanning best to worst error | model input images",fontsize=13)
    fig.savefig(args.output/"tactile_prediction_examples.png",dpi=140);plt.close(fig)
    report = {"status": "complete", "checkpoint": str(args.checkpoint.resolve()), "fresh_seed_check": "passed",
              "prediction_dimension": 1, **metrics}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print("NORMAL_EVAL_REPORT "+json.dumps(report))


if __name__ == "__main__": main()
