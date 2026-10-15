"""Frozen, calibrated v4 stereo angle estimator for control-time inference."""
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from tactile_calibration import require_matching_calibration
from .normal_learning import CHECKPOINT_SCHEMA, NormalAngleNet


class FrozenNormalEstimator:
    def __init__(self, checkpoint, calibration, device):
        path = Path(checkpoint).resolve(strict=True)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Expected a v4 scalar angle checkpoint (not an RL policy)")
        require_matching_calibration(ck["tactile_calibration"], calibration)
        zero = torch.tensor(ck["target_contract"]["zero_rotation_to"])
        if not torch.allclose(zero, torch.diag(torch.tensor([-1., 1., -1.])), atol=1e-6):
            raise ValueError("Checkpoint zero frame must be RotY(pi)")
        if ck["target_contract"]["angle_range_deg"] != [-15., 15.]:
            raise ValueError("This task expects the calibrated +/-15 degree model")
        self.model = NormalAngleNet(**ck["config"]["model"]).to(device).eval()
        self.model.load_state_dict(ck["model_state"])
        self.model.requires_grad_(False)
        self.image_size = tuple(ck["config"]["image_size_hw"])
        self.metadata = {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "calibration_id": calibration["id"], "target_contract": ck["target_contract"],
                         "input_fields": ["left_rgb", "right_rgb", "actual_finger_joint_sum_m"]}
        if "dynamic_adaptation" in ck:
            self.metadata["dynamic_adaptation"] = ck["dynamic_adaptation"]
            self.metadata["dynamic_validation_metrics"] = ck["validation_metrics"]
        self.calls = 0

    @torch.inference_mode()
    def __call__(self, tactile, opening, min_pixels=64, min_depth_mm=.02):
        images, valid = [], torch.isfinite(opening) & (opening >= 0)
        for side in ("left", "right"):
            rgb, depth = tactile[side]["rgb"], tactile[side]["press_depth_mm"]
            if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
                raise ValueError("Expected batched native uint8 NHWC RGB")
            images.append(F.interpolate(rgb.permute(0, 3, 1, 2).float()/255., size=self.image_size,
                                        mode="bilinear", align_corners=False, antialias=True))
            valid &= ((depth > min_depth_mm).sum((1, 2)) >= min_pixels)
            valid &= torch.isfinite(depth).all(dim=(1, 2)) & (rgb.sum(dim=(1, 2, 3)) > 0)
        angle_deg = self.model(*images, opening)
        valid &= torch.isfinite(angle_deg)
        self.calls += 1
        return torch.where(valid, angle_deg, torch.zeros_like(angle_deg)), valid
