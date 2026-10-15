"""Stereo RGB + actual opening -> one signed in-hand angle (degrees)."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tactile_calibration import dataset_calibration, require_matching_calibration
from .data import prepare_image
from .normal_sampling import SCHEMA

CHECKPOINT_SCHEMA = "tactile_normal_angle_checkpoint_v4"


class NormalAngleNet(nn.Module):
    def __init__(self, feature_dim=128, angle_scale_deg=15., opening_scale_m=.08):
        super().__init__()
        layers = []
        for index, (a, b) in enumerate(zip((3, 24, 48, 96), (24, 48, 96, 128))):
            k = 5 if index == 0 else 3
            layers += [nn.Conv2d(a, b, k, stride=2, padding=k//2), nn.GroupNorm(8, b), nn.ReLU(inplace=True)]
        layers += [nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(), nn.Linear(128*3*4, feature_dim), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*layers)
        self.fusion = nn.Sequential(nn.Linear(2*feature_dim+1, 256), nn.ReLU(inplace=True),
                                    nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.register_buffer("angle_scale_deg", torch.tensor(float(angle_scale_deg)))
        self.register_buffer("opening_scale_m", torch.tensor(float(opening_scale_m)))
        nn.init.normal_(self.fusion[-1].weight, std=1e-4)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, left, right, opening):
        if left.shape != right.shape or left.ndim != 4 or left.shape[1] != 3:
            raise ValueError("Two equally sized BCHW RGB tensors are required")
        n = left.shape[0]
        features = self.encoder(torch.cat((left, right), dim=0) * 2 - 1)
        inputs = torch.cat((features[:n], features[n:], opening.reshape(n, 1)/self.opening_scale_m*2-1), dim=1)
        return torch.tanh(self.fusion(inputs)[:, 0]) * self.angle_scale_deg


def resolve_paths(inputs):
    result = []
    for item in inputs:
        item = Path(item).resolve(strict=True)
        if item.is_file(): result.append(item)
        elif (item / "collection.json").exists():
            index = json.loads((item / "collection.json").read_text())
            if index["status"] != "complete": raise ValueError(f"Incomplete collection: {item}")
            result.extend(Path(d["path"]).resolve(strict=True) for d in index["datasets"])
        else: result.append((item / "samples.h5").resolve(strict=True))
    return result


def read_manifest(inputs):
    paths = resolve_paths(inputs)
    records, fingerprints, uuids, seeds = [], [], set(), set()
    calibration = contract = None
    for i, path in enumerate(paths):
        with h5py.File(path) as f:
            if f.attrs.get("schema") != SCHEMA or f.attrs.get("status") != "complete":
                raise ValueError(f"Expected complete upright single-normal v4 data: {path}")
            meta = json.loads(f.attrs["metadata_json"])
            current = dataset_calibration(f)
            if calibration is not None: require_matching_calibration(calibration, current)
            calibration = current
            current_contract = {k: meta[k] for k in ("pose_constraint", "angle_convention", "zero_rotation_to", "object_yaw_deg", "angle_range_deg")}
            if contract is not None and current_contract != contract: raise ValueError("Angle/pose convention mismatch")
            contract = current_contract
            uid = str(f.attrs["dataset_uuid"])
            if uid in uuids: raise ValueError("Duplicate collection UUID")
            uuids.add(uid)
            valid = np.flatnonzero(f["quality/sample_valid"][:])
            all_seeds = set(map(int, f["random/seed"][:]))
            if seeds & all_seeds: raise ValueError("Repeated sampling seeds across datasets")
            seeds |= all_seeds
            for row in valid:
                records.append({"file": i, "row": int(row),
                    "group": f"{uid}:{int(f['ids/contact_pose_id'][row])}",
                    "object_id": int(f["ids/object_id"][row]), "angle_bin": int(f["ids/angle_bin"][row]),
                    "seed": int(f["random/seed"][row])})
            fingerprints.append({"uuid": uid, "samples": len(f["quality/sample_valid"]),
                                 "valid_samples": len(valid), "calibration_id": current["id"]})
    return {"schema": "tactile_normal_split_v4", "files": list(map(str, paths)), "datasets": fingerprints,
            "tactile_calibration": calibration, "target_contract": contract, "records": records}


def split_manifest(inputs, seed=0):
    manifest = read_manifest(inputs)
    cells = {}
    for record in manifest.pop("records"):
        cells.setdefault((record["object_id"], record["angle_bin"]), []).append(record)
    splits = {key: [] for key in ("train", "validation", "test")}
    rng = np.random.default_rng(seed)
    for cell, records in sorted(cells.items()):
        if len(records) < 6: raise ValueError(f"Need >=6 independent poses per object/angle bin, got {len(records)} at {cell}")
        rng.shuffle(records)
        n = max(1, round(len(records)*.15))
        splits["validation"] += records[:n]; splits["test"] += records[n:2*n]; splits["train"] += records[2*n:]
    for split in splits:
        splits[split].sort(key=lambda r: (r["file"], r["row"]))
    manifest.update(seed=seed, splits=splits, split_protocol="70/15/15 independent poses, stratified by object and angle bin")
    return manifest


def fresh_manifest(inputs, training):
    manifest = read_manifest(inputs)
    require_matching_calibration(training["tactile_calibration"], manifest["tactile_calibration"])
    if training["target_contract"] != manifest["target_contract"]: raise ValueError("Angle convention mismatch")
    old_ids = {d["uuid"] for d in training["datasets"]}
    if old_ids & {d["uuid"] for d in manifest["datasets"]}: raise ValueError("Fresh test requires new dataset UUIDs")
    old_seeds = set()
    for path in training["files"]:
        with h5py.File(path) as f: old_seeds.update(map(int, f["random/seed"][:]))
    for path in manifest["files"]:
        with h5py.File(path) as f:
            if old_seeds & set(map(int, f["random/seed"][:])): raise ValueError("Fresh test seeds overlap training collection")
    manifest["splits"] = {"external": manifest.pop("records")}
    return manifest


class AngleDataset(Dataset):
    def __init__(self, manifest, split, image_size=(120, 160)):
        self.records = manifest["splits"][split]
        self.cache = []
        handles = [h5py.File(path) for path in manifest["files"]]
        try:
            for f, expected in zip(handles, manifest["datasets"]):
                if (f.attrs["dataset_uuid"] != expected["uuid"] or f.attrs["status"] != "complete" or
                        len(f["quality/sample_valid"]) != expected["samples"]):
                    raise ValueError("Dataset changed since split construction")
            for record in self.records:
                f, row = handles[record["file"]], record["row"]
                self.cache.append({
                    "left": prepare_image(f["tactile/left/rgb"][row], image_size),
                    "right": prepare_image(f["tactile/right/rgb"][row], image_size),
                    "opening": torch.from_numpy(f["gripper/opening_joint_sum_m"][row].astype(np.float32)),
                    "angle": torch.tensor(float(f["target/in_hand_angle_deg"][row]), dtype=torch.float32),
                    "object_id": record["object_id"], "row": row,
                })
        finally:
            for f in handles: f.close()

    def __len__(self): return len(self.cache)
    def __getitem__(self, index): return self.cache[index]


def angle_metrics(prediction, target):
    delta = np.asarray(prediction) - target
    error = np.abs(delta)
    return {"samples": len(error), "mae_deg": float(error.mean()), "rmse_deg": float(np.sqrt((delta**2).mean())),
            "median_deg": float(np.median(error)), "p95_deg": float(np.quantile(error, .95)),
            "max_deg": float(error.max()), "bias_deg": float(delta.mean()),
            "within_1_deg": float((error <= 1).mean()), "within_2_deg": float((error <= 2).mean()),
            "within_5_deg": float((error <= 5).mean())}


@torch.inference_mode()
def evaluate_angles(model, loader, device, baseline):
    model.eval()
    prediction, target, objects = [], [], []
    for batch in loader:
        prediction.append(model(batch["left"].to(device), batch["right"].to(device), batch["opening"].to(device)).cpu().numpy())
        target.append(batch["angle"].numpy()); objects.append(batch["object_id"].numpy())
    prediction, target, objects = map(np.concatenate, (prediction, target, objects))
    result = angle_metrics(prediction, target)
    result["baseline_train_mean"] = angle_metrics(np.full_like(target, baseline), target)
    result["per_object"] = {str(obj): angle_metrics(prediction[objects == obj], target[objects == obj]) for obj in np.unique(objects)}
    return result, {"prediction_deg": prediction, "target_deg": target, "object_id": objects,
                    "error_deg": prediction-target}
