"""HDF5 dataset and splits by independent grasp; simulator ground truth is label-only."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tactile_calibration import dataset_calibration, require_matching_calibration


def prepare_image(rgb, image_size=(120, 160)):
    image = torch.as_tensor(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 255.0
    return F.interpolate(image[None], size=image_size, mode="bilinear", align_corners=False,
                         antialias=True)[0]


def split_groups(groups, seed=0, validation_fraction=0.15, test_fraction=0.15):
    keys = sorted(set(groups))
    if len(keys) < 3:
        raise ValueError("Need at least 3 independent valid grasps for train/validation/test")
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1 and validation_fraction + test_fraction < 1):
        raise ValueError("Invalid split fractions")
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, round(len(keys) * validation_fraction))
    n_test = max(1, round(len(keys) * test_fraction))
    if n_val + n_test >= len(keys):
        raise ValueError("Not enough training grasps after splitting")
    return {"train": sorted(keys[n_val+n_test:]), "validation": sorted(keys[:n_val]),
            "test": sorted(keys[n_val:n_val+n_test])}


def make_manifest(paths, seed=0, validation_fraction=0.15, test_fraction=0.15):
    files, records, fingerprints, seen = [], [], [], set()
    calibration = None
    for file_id, path in enumerate(paths):
        path = Path(path).resolve(strict=True)
        with h5py.File(path, "r") as f:
            if f.attrs.get("schema") not in ("tactile_pose_v1", "tactile_pose_v2") or f.attrs.get("status") != "complete":
                raise ValueError(f"Not a complete pose dataset: {path}")
            current_calibration = dataset_calibration(f)
            if calibration is not None:
                require_matching_calibration(calibration, current_calibration)
            calibration = current_calibration
            dataset_uuid = str(f.attrs["dataset_uuid"])
            if dataset_uuid in seen:
                raise ValueError("Duplicate dataset UUID: do not include copies of the same collection")
            seen.add(dataset_uuid)
            valid = np.flatnonzero(f["quality/sample_valid"][:])
            grasp_ids = f["ids/grasp_id"][:]
            for row in valid:
                records.append({"file": file_id, "row": int(row), "group": f"{dataset_uuid}:{int(grasp_ids[row])}"})
            fingerprints.append({"uuid": dataset_uuid, "samples": len(grasp_ids), "valid_samples": len(valid),
                                 "calibration_id": calibration["id"]})
        files.append(str(path))
    assignments = split_groups([record["group"] for record in records], seed, validation_fraction, test_fraction)
    assignment = {group: split for split, groups in assignments.items() for group in groups}
    splits = {name: [] for name in assignments}
    for record in records:
        splits[assignment[record["group"]]].append(record)
    return {"schema": "tactile_pose_split_v1", "seed": seed, "files": files, "datasets": fingerprints,
            "groups": assignments, "splits": splits, "filter": "quality/sample_valid == true",
            "tactile_calibration": calibration}


def external_manifest(paths, training_manifest):
    """Evaluate new collections only, with no overlapping data UUID or seed."""
    manifest = make_manifest(paths)
    require_matching_calibration(training_manifest.get("tactile_calibration", {}), manifest["tactile_calibration"])
    old_ids = {d["uuid"] for d in training_manifest["datasets"]}
    if old_ids & {d["uuid"] for d in manifest["datasets"]}:
        raise ValueError("External evaluation requires new collection UUIDs")
    previous_seeds = set()
    for path in training_manifest["files"]:
        with h5py.File(path, "r") as f:
            previous_seeds.update(map(int, f["random/seed"][:]))
    for path in manifest["files"]:
        with h5py.File(path, "r") as f:
            if previous_seeds & set(map(int, f["random/seed"][:])):
                raise ValueError("External evaluation seeds overlap the training collection")
    records = sorted([r for rows in manifest["splits"].values() for r in rows], key=lambda r: (r["file"], r["row"]))
    manifest["splits"] = {"external": records}
    manifest["groups"] = {"external": sorted({r["group"] for r in records})}
    return manifest


class PoseDataset(Dataset):
    def __init__(self, manifest, split, image_size=(120, 160)):
        self.files = manifest["files"]
        self.records = manifest["splits"][split]
        self.image_size = tuple(image_size)
        self.handles = {}
        self.cached_samples = None
        for path, expected in zip(self.files, manifest["datasets"]):
            with h5py.File(path, "r") as f:
                if (f.attrs.get("status") != "complete" or
                        f.attrs.get("dataset_uuid") != expected["uuid"] or
                        len(f["ids/grasp_id"]) != expected["samples"]):
                    raise ValueError(f"Dataset no longer matches the saved split: {path}")
                if "calibration_id" in expected:
                    require_matching_calibration({"id": expected["calibration_id"]}, dataset_calibration(f))

    def __len__(self):
        return len(self.records)

    def file(self, index):
        if index not in self.handles:
            self.handles[index] = h5py.File(self.files[index], "r")
        return self.handles[index]

    def __getitem__(self, index):
        if self.cached_samples is not None:
            return self.cached_samples[index]
        record = self.records[index]
        f = self.file(record["file"])
        row = record["row"]
        return {
            "left": prepare_image(f["tactile/left/rgb"][row], self.image_size),
            "right": prepare_image(f["tactile/right/rgb"][row], self.image_size),
            "opening": torch.from_numpy(f["gripper/opening_joint_sum_m"][row].astype(np.float32)),
            "rotation": torch.from_numpy(f["target/rotation_go"][row].astype(np.float32)),
            "index": index,
        }

    def preload(self):
        """Cache the identical resized float tensors in RAM; preserve row order."""
        if self.cached_samples is None:
            self.cached_samples = [self[index] for index in range(len(self))]

    def labels(self):
        if self.cached_samples is not None:
            return torch.stack([sample["rotation"] for sample in self.cached_samples])
        return torch.stack([torch.from_numpy(self.file(r["file"])["target/rotation_go"][r["row"]].astype(np.float32))
                            for r in self.records])

    def close(self):
        for file in self.handles.values():
            file.close()
        self.handles.clear()
        self.cached_samples = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handles"] = {}
        return state
