"""Evaluate a saved checkpoint on its fixed, grasp-disjoint held-out split."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import DataLoader

from tactile_pose.data import PoseDataset, external_manifest
from tactile_pose.evaluation import evaluate, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--data", type=Path, nargs="+", help="Optional fresh collection(s), disjoint UUIDs and seeds; evaluate all valid rows")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, ckpt = load_model(args.checkpoint, args.device)
    manifest = external_manifest(args.data, ckpt["manifest"]) if args.data else ckpt["manifest"]
    split = "external" if args.data else args.split
    data = PoseDataset(manifest, split, ckpt["image_size_hw"])
    try:
        loader = DataLoader(data, batch_size=32, shuffle=False)
        metrics, prediction, errors = evaluate(model, loader, args.device, ckpt["baseline_rotation"])
        quaternion = Rotation.from_matrix(prediction).as_quat()[:, [3, 0, 1, 2]]
        np.savez_compressed(args.output / "predictions.npz", rotation_go=prediction,
                            quaternion_go_wxyz=quaternion, error_deg=errors,
                            target_rotation_go=data.labels().numpy())
        with (args.output / "predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source_h5", "source_row", "grasp_group", "qw", "qx", "qy", "qz", "error_deg"])
            for record, q, error in zip(data.records, quaternion, errors):
                writer.writerow([data.files[record["file"]], record["row"], record["group"], *q, error])
        report = {"checkpoint": str(args.checkpoint.resolve()), "split": split, **metrics,
                  "tactile_calibration": manifest.get("tactile_calibration", {"id": "legacy_shared_28mm"}),
                  "source_files": data.files}
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        data.close()


if __name__ == "__main__":
    main()
