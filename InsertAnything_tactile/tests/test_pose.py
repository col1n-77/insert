import numpy as np
import h5py
import pytest
import torch
from scipy.spatial.transform import Rotation

from tactile_pose.rotations import (quaternion_to_matrix, relative_pose, rotation_6d_to_matrix,
                                    matrix_to_rotation_6d, geodesic_angle)
from tactile_pose.data import split_groups, make_manifest
from tactile_pose.model import TactileRotationNet

torch.set_num_threads(2)


def test_relative_pose_matches_independent_scipy_and_point_transform():
    rng = np.random.default_rng(9)
    g = Rotation.random(20, random_state=rng)
    o = Rotation.random(20, random_state=rng)
    pg, po = rng.normal(size=(2, 20, 3))
    qg = torch.tensor(g.as_quat()[:, [3, 0, 1, 2]])
    qo = torch.tensor(o.as_quat()[:, [3, 0, 1, 2]])
    p, r = relative_pose(torch.tensor(pg), qg, torch.tensor(po), qo)
    np.testing.assert_allclose(r, (g.inv() * o).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(p, g.inv().apply(po - pg), atol=1e-12)
    point = rng.normal(size=(20, 3))
    expected = g.inv().apply(o.apply(point) + po - pg)
    actual = (r @ torch.tensor(point)[..., None]).squeeze(-1) + p
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    torch.testing.assert_close(quaternion_to_matrix(qo), quaternion_to_matrix(-qo))


def test_known_wxyz_rotation():
    q = torch.tensor([[2**-0.5, 0, 0, 2**-0.5]], dtype=torch.float64)
    r = quaternion_to_matrix(q)[0]
    torch.testing.assert_close(r @ torch.tensor([1., 0., 0.], dtype=torch.float64),
                              torch.tensor([0., 1., 0.], dtype=torch.float64), atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        quaternion_to_matrix(torch.zeros(4))


def test_6d_column_convention_and_so3_even_when_degenerate():
    matrices = torch.tensor(Rotation.random(20, random_state=0).as_matrix())
    torch.testing.assert_close(rotation_6d_to_matrix(matrix_to_rotation_6d(matrices)), matrices)
    values = torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 0., 0., 2., 0., 0.], [0., 2., 0., 0., 4., 0.]])
    r = rotation_6d_to_matrix(values)
    torch.testing.assert_close(r.transpose(-1, -2) @ r, torch.eye(3).expand(3, 3, 3))
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(3))


def test_geodesic_angles_and_finite_gradients():
    matrices = torch.tensor(Rotation.from_euler("z", [0, 90, 180], degrees=True).as_matrix())
    angles = torch.rad2deg(geodesic_angle(torch.eye(3, dtype=torch.float64), matrices))
    torch.testing.assert_close(angles, torch.tensor([0., 90., 180.], dtype=torch.float64))
    values = matrix_to_rotation_6d(matrices).clone().requires_grad_()
    geodesic_angle(rotation_6d_to_matrix(values), matrices).mean().backward()
    assert torch.isfinite(values.grad).all()


def test_split_disjoint_by_grasp_not_by_frame():
    groups = [f"run:{i}" for i in range(20) for _ in range(8)]
    splits = split_groups(groups, seed=42)
    assert splits == split_groups(groups, seed=42)
    sets = [set(splits[key]) for key in ("train", "validation", "test")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert set.union(*sets) == set(groups)
    with pytest.raises(ValueError):
        split_groups(["same_grasp"] * 100)


def test_manifest_filters_invalid_samples_and_rejects_dataset_copies(tmp_path):
    path = tmp_path / "samples.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "tactile_pose_v1"
        f.attrs["status"] = "complete"
        f.attrs["dataset_uuid"] = "test-run"
        f["ids/grasp_id"] = np.repeat(np.arange(6), 3)
        valid = np.ones(18, dtype=bool)
        valid[3:6] = False
        valid[8] = False
        f["quality/sample_valid"] = valid
    manifest = make_manifest([path], seed=0)
    records = [r for subset in manifest["splits"].values() for r in subset]
    assert sorted(r["row"] for r in records) == np.flatnonzero(valid).tolist()
    owners = {}
    for split, rows in manifest["splits"].items():
        for record in rows:
            assert owners.setdefault(record["group"], split) == split
    with pytest.raises(ValueError, match="Duplicate dataset UUID"):
        make_manifest([path, path])


def test_stereo_model_gradient_reaches_both_images_and_opening():
    torch.manual_seed(2)
    model = TactileRotationNet()
    left = torch.rand(2, 3, 120, 160, requires_grad=True)
    right = torch.rand(2, 3, 120, 160, requires_grad=True)
    opening = torch.tensor([[0.005], [0.008]], requires_grad=True)
    output = model(left, right, opening)
    target = torch.tensor(Rotation.from_euler("xyz", [[10, 20, 30], [20, 5, 40]], degrees=True).as_matrix(), dtype=torch.float32)
    geodesic_angle(output, target).mean().backward()
    for tensor in (left, right, opening):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    torch.testing.assert_close(output.transpose(-1, -2) @ output, torch.eye(3).expand(2, 3, 3), atol=1e-5, rtol=1e-5)
    assert sum(p.numel() for p in model.parameters()) < 500_000
