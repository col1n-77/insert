"""Train the v1 stereo CNN with a grasp-disjoint validation/test protocol."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from tactile_pose.data import PoseDataset, make_manifest
from tactile_pose.evaluation import evaluate
from tactile_pose.model import TactileRotationNet
from tactile_pose.rotations import geodesic_angle, mean_rotation

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pose_v1.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for name in ("epochs", "batch_size", "seed"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if min(config["epochs"], config["batch_size"]) < 1:
        parser.error("epochs and batch_size must be positive")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = (args.output or ROOT / "outputs" / datetime.now().strftime("pose_train_%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = make_manifest(args.data, seed, config["validation_fraction"], config["test_fraction"])
    (output / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    datasets = {name: PoseDataset(manifest, name, config["image_size_hw"]) for name in manifest["splits"]}
    if config.get("cache_images", False):
        for name, dataset in datasets.items():
            dataset.preload()
            print(f"POSE_CACHE {name} samples={len(dataset)}", flush=True)
    loaders = {name: DataLoader(dataset, batch_size=config["batch_size"], shuffle=(name == "train"),
                              num_workers=0) for name, dataset in datasets.items()}
    # Evaluation order must match the manifest, including for training metrics.
    train_eval_loader = DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=False)
    baseline = mean_rotation(datasets["train"].labels())
    model = TactileRotationNet(**config["model"])
    model.initialize_rotation(baseline)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config.get("min_learning_rate", 0))
        if config.get("scheduler") == "cosine" else None)
    best = float("inf")
    best_epoch = None
    try:
        with (output / "history.jsonl").open("w") as history:
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                total, count = 0.0, 0
                for batch in loaders["train"]:
                    prediction = model(batch["left"].to(args.device), batch["right"].to(args.device),
                                       batch["opening"].to(args.device))
                    loss = geodesic_angle(prediction, batch["rotation"].to(args.device)).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite rotation loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach()) * len(prediction)
                    count += len(prediction)
                validation, _, _ = evaluate(model, loaders["validation"], args.device, baseline)
                row = {"epoch": epoch, "training_loss_rad": total / count, "validation": validation,
                       "learning_rate": optimizer.param_groups[0]["lr"]}
                history.write(json.dumps(row) + "\n")
                history.flush()
                checkpoint = {
                    "schema": "tactile_rotation_checkpoint_v1", "epoch": epoch,
                    "model_config": config["model"], "image_size_hw": config["image_size_hw"],
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "training_config": config, "manifest": manifest,
                    "tactile_calibration": manifest["tactile_calibration"],
                    "baseline_rotation": baseline.cpu(), "validation_metrics": validation,
                    "input_fields": ["tactile/left/rgb", "tactile/right/rgb", "gripper/opening_joint_sum_m"],
                    "target": "R_GO = R_WG.T @ R_WO", "quaternion_order": "wxyz",
                }
                torch.save(checkpoint, output / "last.pt")
                if validation["mean_deg"] < best:
                    best = validation["mean_deg"]
                    best_epoch = epoch
                    torch.save(checkpoint, output / "best.pt")
                print(f"POSE_TRAIN epoch={epoch}/{config['epochs']} loss_rad={total/count:.6f} "
                      f"val_deg={validation['mean_deg']:.3f} best={best:.3f}", flush=True)
                if scheduler is not None:
                    scheduler.step()
        selected = torch.load(output / "best.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(selected["model_state"])
        train_metrics, _, _ = evaluate(model, train_eval_loader, args.device, baseline)
        validation, _, _ = evaluate(model, loaders["validation"], args.device, baseline)
        # The held-out test set is evaluated only after model selection.
        test, prediction, errors = evaluate(model, loaders["test"], args.device, baseline)
        np.savez_compressed(output / "test_predictions.npz", rotation_go=prediction, error_deg=errors,
                            target_rotation_go=datasets["test"].labels().numpy())
        report = {"status": "complete", "best_epoch": best_epoch, "device": args.device,
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "split_grasps": {name: len(groups) for name, groups in manifest["groups"].items()},
                  "split_samples": {name: len(data) for name, data in datasets.items()},
                  "train": train_metrics, "validation": validation, "test": test,
                  "checkpoint": str(output / "best.pt"),
                  "tactile_calibration": manifest["tactile_calibration"],
                  "scope": "Simulation-only L-hole prototype; small-data metrics do not establish full-SO(3) or real-world generalization"}
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print("POSE_TRAIN_REPORT " + json.dumps(report), flush=True)
        print(f"OUTPUT {output}", flush=True)
    finally:
        for data in datasets.values():
            data.close()


if __name__ == "__main__":
    main()
