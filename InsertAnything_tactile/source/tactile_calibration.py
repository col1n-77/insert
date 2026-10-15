"""Versioned optical reference shared by runtime sensors and offline datasets.

The front plane is a geometric reference for rigid gelpads, not a force or
deformable-gel calibration. Camera clipping stays independent of optical zero.
"""
import hashlib
import json
import math

import torch

LEGACY_ID = "legacy_shared_28mm"


def height_from_depth_m(depth, zero_mm, gel_height_mm, clipping_range_m):
    """Return negative indentation in mm; invalid/out-of-frustum rays are empty."""
    near, far = clipping_range_m
    if not (0 < near < zero_mm / 1000 < far and gel_height_mm > 0):
        raise ValueError("Optical zero must lie inside the camera clipping range")
    valid = torch.isfinite(depth) & (depth >= near) & (depth <= far)
    height = (depth * 1000.0 - zero_mm).clamp(-gel_height_mm, 0.0)
    return torch.where(valid, height, torch.zeros_like(height))


def geometry_zero_mm(stage, robot_path, side):
    # Use USD for BOTH transforms. Runtime Fabric body poses cannot be mixed
    # with camera metadata, which may still contain the authored USD pose.
    from pxr import Gf, Usd, UsdGeom
    gel = stage.GetPrimAtPath(f"{robot_path}/gelpad_{side}")
    camera = stage.GetPrimAtPath(f"{robot_path}/gelsight_mini_case_{side}/Camera")
    if not gel.IsValid() or not camera.IsValid():
        raise ValueError(f"Missing gelpad/camera for {side}")
    cache = UsdGeom.XformCache()
    inverse_camera = cache.GetLocalToWorldTransform(camera).GetInverse()
    depths = []
    for prim in Usd.PrimRange(gel):
        if prim.IsA(UsdGeom.Mesh):
            transform = cache.GetLocalToWorldTransform(prim) * inverse_camera
            for point in UsdGeom.Mesh(prim).GetPointsAttr().Get():
                depths.append(-transform.Transform(Gf.Vec3d(*map(float, point)))[2] * 1000.0)
    if not depths or not math.isfinite(max(depths)) or not 20 < max(depths) < 35:
        raise ValueError(f"Implausible camera-to-gel geometry: {side}, {depths[:3]}")
    # Sub-nanometer transform roundoff should not create a new calibration ID.
    return round(max(depths), 6)


def make_calibration(zero_mm, clipping_range_m, gel_height_mm, resolution_wh, mode):
    if mode not in ("geometry_per_side", "legacy_shared"):
        raise ValueError(f"Unknown tactile calibration mode: {mode}")
    spec = {"schema": "tactile_optical_calibration_v1", "mode": mode,
            "zero_reference_mm": dict(zero_mm), "clipping_range_m": list(clipping_range_m),
            "gel_height_mm": float(gel_height_mm), "resolution_wh": list(resolution_wh),
            "reference": "USD gel front plane in each camera's image-depth coordinates",
            "optics": "Taxim", "with_shadow": False, "image_orientation": "native_each_camera"}
    spec["id"] = (LEGACY_ID if mode == "legacy_shared" else
                  hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest())
    return spec


def dataset_calibration(h5file):
    schema = h5file.attrs.get("schema")
    metadata = json.loads(h5file.attrs.get("metadata_json", "{}"))
    calibration = metadata.get("tactile_calibration")
    if calibration is None and schema == "tactile_pose_v1":
        return {"id": LEGACY_ID, "mode": "legacy_shared", "zero_reference_mm": {"left": 28.0, "right": 28.0}}
    if not calibration or not calibration.get("id"):
        raise ValueError("Dataset is missing its optical calibration")
    return calibration


def require_matching_calibration(expected, actual):
    if expected.get("id", LEGACY_ID) != actual["id"]:
        raise ValueError(f"Optical calibration mismatch: checkpoint/data {expected.get('id', LEGACY_ID)} vs {actual['id']}")
