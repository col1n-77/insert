import json

import h5py
import numpy as np
import pytest
import torch

from tactile_calibration import height_from_depth_m, dataset_calibration, require_matching_calibration
from tactile_pose.data import make_manifest, PoseDataset, external_manifest


def test_equal_physical_indentation_despite_different_optical_mounts():
    for zero in (28.246235, 27.950882):
        depth = torch.tensor([zero, zero - 0.25, zero + 0.25]) / 1000
        actual = height_from_depth_m(depth, zero, 4.0, (0.024, 0.034))
        torch.testing.assert_close(actual, torch.tensor([0., -0.25, 0.]), atol=3e-6, rtol=0)
    invalid = torch.tensor([float("nan"), float("inf"), -float("inf"), 0., -1., 0.05])
    assert height_from_depth_m(invalid, 28.0, 4.0, (0.024, 0.034)).eq(0).all()
    with pytest.raises(ValueError):
        height_from_depth_m(torch.tensor([0.028]), 40.0, 4.0, (0.024, 0.034))


def make_file(path, uid, calibration=None):
    with h5py.File(path, "w") as f:
        f.attrs.update(schema="tactile_pose_v2" if calibration else "tactile_pose_v1", status="complete", dataset_uuid=uid)
        if calibration:
            f.attrs["metadata_json"] = json.dumps({"tactile_calibration": calibration})
        f["ids/grasp_id"] = np.arange(9)
        f["quality/sample_valid"] = np.ones(9, dtype=bool)


def test_prevent_mixing_or_silently_replacing_calibration(tmp_path):
    old, new = tmp_path / "old.h5", tmp_path / "new.h5"
    make_file(old, "old")
    make_file(new, "new", {"id": "per_side_test"})
    with pytest.raises(ValueError, match="calibration mismatch"):
        make_manifest([old, new])
    manifest = make_manifest([new])
    data = PoseDataset(manifest, "test")
    data.close()
    with h5py.File(new, "r+") as f:
        f.attrs["metadata_json"] = json.dumps({"tactile_calibration": {"id": "changed"}})
    with pytest.raises(ValueError, match="calibration mismatch"):
        PoseDataset(manifest, "test")
    with h5py.File(new, "r") as f:
        with pytest.raises(ValueError, match="calibration mismatch"):
            require_matching_calibration({}, dataset_calibration(f))


def test_v2_requires_calibration_and_v1_remains_readable(tmp_path):
    path = tmp_path / "data.h5"
    make_file(path, "old")
    assert make_manifest([path])["tactile_calibration"]["id"] == "legacy_shared_28mm"
    with h5py.File(path, "r+") as f:
        f.attrs["schema"] = "tactile_pose_v2"
    with pytest.raises(ValueError, match="missing.*calibration"):
        make_manifest([path])


def test_external_evaluation_rejects_reused_collections_and_seeds(tmp_path):
    train, fresh = tmp_path / "train.h5", tmp_path / "fresh.h5"
    for path, uid, seed in ((train, "train", 20), (fresh, "fresh", 100)):
        make_file(path, uid, {"id": "per_side_test"})
        with h5py.File(path, "r+") as f:
            f["random/seed"] = np.arange(seed, seed + 9)
    training = make_manifest([train])
    external = external_manifest([fresh], training)
    assert len(external["splits"]["external"]) == 9
    with pytest.raises(ValueError, match="new collection UUIDs"):
        external_manifest([train], training)
    with h5py.File(fresh, "r+") as f:
        f["random/seed"][0] = 20
    with pytest.raises(ValueError, match="seeds overlap"):
        external_manifest([fresh], training)
