"""Offline inference, geodesic metrics, and a training-mean baseline."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .model import TactileRotationNet
from .rotations import geodesic_angle


def summarize(errors_deg, records):
    values = np.asarray(errors_deg, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("No finite evaluation errors")
    group_errors = {}
    for value, record in zip(values, records):
        group_errors.setdefault(record["group"], []).append(float(value))
    return {"samples": len(values), "grasps": len(group_errors),
            "mean_deg": float(values.mean()), "median_deg": float(np.median(values)),
            "rmse_deg": float(np.sqrt(np.mean(values ** 2))), "p95_deg": float(np.percentile(values, 95)),
            "within_1_deg": float(np.mean(values <= 1)), "within_3_deg": float(np.mean(values <= 3)),
            "within_5_deg": float(np.mean(values <= 5)), "within_10_deg": float(np.mean(values <= 10)),
            "mean_per_grasp_deg": float(np.mean([np.mean(v) for v in group_errors.values()]))}


@torch.inference_mode()
def evaluate(model, loader, device, baseline=None):
    model.eval()
    errors, baseline_errors, predictions = [], [], []
    for batch in loader:
        rotation = model(batch["left"].to(device), batch["right"].to(device), batch["opening"].to(device))
        target = batch["rotation"].to(device)
        errors.extend(torch.rad2deg(geodesic_angle(rotation, target)).cpu().tolist())
        predictions.append(rotation.cpu().numpy())
        if baseline is not None:
            baseline_errors.extend(torch.rad2deg(geodesic_angle(baseline.to(device), target)).cpu().tolist())
    metrics = summarize(errors, loader.dataset.records)
    prediction_array = np.concatenate(predictions)
    targets = loader.dataset.labels().numpy()
    # Residual axes are expressed in the gripper frame. Their vector norm is
    # the SO(3) error; unlike Euler differences, these have no +/-180 wrap.
    residual = Rotation.from_matrix(prediction_array @ targets.transpose(0, 2, 1)).as_rotvec(degrees=True)
    metrics["residual_rotvec_mae_xyz_deg"] = np.abs(residual).mean(0).tolist()
    if baseline is not None:
        metrics["training_mean_baseline"] = summarize(baseline_errors, loader.dataset.records)
    return metrics, prediction_array, np.asarray(errors)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TactileRotationNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
