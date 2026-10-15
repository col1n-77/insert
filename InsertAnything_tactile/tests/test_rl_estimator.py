import importlib.util
from pathlib import Path
import h5py
import numpy as np
import pytest
import torch
from tactile_pose.data import prepare_image
from tactile_pose.normal_learning import NormalAngleNet
from tactile_pose.rl_estimator import FrozenNormalEstimator


def test_online_preprocessing_matches_offline_and_masks_empty_contact():
    torch.manual_seed(23)
    estimator = FrozenNormalEstimator.__new__(FrozenNormalEstimator)
    estimator.model = NormalAngleNet().eval().requires_grad_(False)
    estimator.image_size = (120, 160); estimator.calls = 0
    data = {}
    for side in ("left", "right"):
        rgb = torch.randint(0, 256, (2, 240, 320, 3), dtype=torch.uint8)
        depth = torch.ones((2, 240, 320))*.3
        depth[1] = 0
        data[side] = {"rgb": rgb, "press_depth_mm": depth}
    opening = torch.tensor([.006, .012])
    prediction, valid = estimator(data, opening)
    offline = [prepare_image(data[s]["rgb"][0].numpy()).unsqueeze(0) for s in ("left", "right")]
    with torch.no_grad(): expected = estimator.model(*offline, opening[:1])
    torch.testing.assert_close(prediction[:1], expected, atol=1e-5, rtol=1e-5)
    assert valid.tolist() == [True, False] and prediction[1] == 0
    assert estimator.calls == 1


def test_physical_grasp_frames_never_cross_splits(tmp_path):
    import json
    path = Path(__file__).resolve().parents[1]/"scripts/adapt_dynamic_angle.py"
    spec = importlib.util.spec_from_file_location("dynamic_adapt", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    h5 = tmp_path/"samples.h5"
    n = 60
    with h5py.File(h5, "w") as f:
        f.attrs.update(schema="tactile_dynamic_normal_v1", status="complete", dataset_uuid="test-grasps")
        f.attrs["metadata_json"] = json.dumps({"seed":31,"object_id":2,"tactile_calibration":{"id":"cal1"}})
        f["quality/sample_valid"] = np.ones(n,dtype=bool)
        f["ids/grasp_id"] = np.repeat(np.arange(20),3)
        f["target/in_hand_angle_deg"] = np.repeat(np.tile([-12.,-6.,0.,6.,12.],4),3)
    ck={"tactile_calibration":{"id":"cal1"}}
    manifest=mod.manifest_for([h5],ck,1)
    groups={key:{row["group"] for row in rows} for key,rows in manifest["splits"].items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert sum(map(len,manifest["splits"].values()))==n
    ck["dynamic_manifest"]=manifest
    with pytest.raises(ValueError,match="new collection UUIDs and seeds"):
        mod.manifest_for([h5],ck,1,external=True)
