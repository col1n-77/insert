"""Train only the grasp-normal in-hand angle from stereo tactile RGB + opening."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tactile_pose.normal_learning import (CHECKPOINT_SCHEMA, NormalAngleNet, AngleDataset,
                                         split_manifest, evaluate_angles)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.seed < 0 or not 0 < args.learning_rate < 1:
        p.error("Invalid training counts/seed/learning rate")
    if args.output.exists(): p.error("Output must be a new directory")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    manifest = split_manifest(args.data, args.seed)
    scale = max(abs(a) for a in manifest["target_contract"]["angle_range_deg"])
    config = {"epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
              "learning_rate": args.learning_rate, "weight_decay": 1e-4,
              "image_size_hw": [120, 160], "model": {"feature_dim": 128, "angle_scale_deg": scale, "opening_scale_m": .08},
              "loss": "MSE on angle/angle_scale_deg", "scheduler": "cosine", "device": args.device}
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n")
    datasets = {key: AngleDataset(manifest, key, config["image_size_hw"]) for key in manifest["splits"]}
    loaders = {key: DataLoader(data, batch_size=args.batch_size, shuffle=key=="train") for key, data in datasets.items()}
    baseline = float(np.mean([float(x["angle"]) for x in datasets["train"].cache]))
    model = NormalAngleNet(**config["model"]).to(args.device)
    with torch.no_grad(): model.fusion[-1].bias.fill_(float(np.arctanh(np.clip(baseline/scale, -.99, .99))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=2e-5)
    best, best_epoch = float("inf"), None
    with (args.output / "history.jsonl").open("x") as history:
        for epoch in range(1, args.epochs+1):
            model.train(); total = count = 0
            for batch in loaders["train"]:
                prediction = model(batch["left"].to(args.device), batch["right"].to(args.device), batch["opening"].to(args.device))
                loss = ((prediction-batch["angle"].to(args.device))/scale).square().mean()
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite angle loss")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step(); total += float(loss.detach())*len(prediction); count += len(prediction)
            validation, _ = evaluate_angles(model, loaders["validation"], args.device, baseline)
            row = {"epoch": epoch, "train_normalized_mse": total/count, "validation": validation,
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            history.write(json.dumps(row)+"\n"); history.flush()
            checkpoint = {"schema": CHECKPOINT_SCHEMA, "epoch": epoch,
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "config": config, "manifest": manifest, "baseline_train_mean_deg": baseline,
                "validation_metrics": validation, "tactile_calibration": manifest["tactile_calibration"],
                "target_contract": manifest["target_contract"], "target": "target/in_hand_angle_deg",
                "input_fields": ["tactile/left/rgb", "tactile/right/rgb", "gripper/opening_joint_sum_m"],
                "output": "one signed angle in degrees about TCP +Y relative to nominal insertion grasp"}
            if validation["mae_deg"] < best:
                best, best_epoch = validation["mae_deg"], epoch
                torch.save(checkpoint, args.output / "best.pt")
            if epoch == args.epochs: torch.save(checkpoint, args.output / "last.pt")
            scheduler.step()
            print(f"NORMAL_TRAIN epoch={epoch}/{args.epochs} loss={total/count:.6f} val_MAE={validation['mae_deg']:.4f} best={best:.4f}", flush=True)
    chosen = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(chosen["model_state"])
    report = {"status": "complete", "best_epoch": best_epoch, "checkpoint": str((args.output / "best.pt").resolve()),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "split_samples": {key: len(d) for key, d in datasets.items()}, "prediction_dimension": 1}
    for name, dataset in datasets.items():
        metrics, arrays = evaluate_angles(model, DataLoader(dataset, batch_size=args.batch_size, shuffle=False), args.device, baseline)
        report[name] = metrics
        if name == "test": np.savez_compressed(args.output / "test_predictions.npz", **arrays)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print("NORMAL_TRAIN_REPORT "+json.dumps(report), flush=True)


if __name__ == "__main__": main()
