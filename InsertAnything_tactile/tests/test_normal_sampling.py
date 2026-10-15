import numpy as np
import h5py
import json
import pytest
import torch
from scipy.spatial.transform import Rotation
from tactile_pose.normal_sampling import (NORMAL_T, ZERO_ROTATION_TO, world_object_rotation,
    normal_rotation_to, tcp_rotation_for_upright_object, measure_normal_angle, propose_normal_pose)
from tactile_pose.tcp_kinematics import TcpKinematics
from tactile_pose.normal_learning import NormalAngleNet, split_manifest, fresh_manifest
from tactile_pose.normal_sampling import SCHEMA


def test_object_vertical_and_relative_rotation_only_about_grasp_normal():
    for yaw in (0, 47, 180):
        for angle in (-15, -7.5, 0, 7.5, 15):
            r_to = normal_rotation_to(angle)
            r_wt = tcp_rotation_for_upright_object(angle, yaw)
            r_wo = r_wt @ r_to
            np.testing.assert_allclose(r_wo, world_object_rotation(yaw), atol=1e-12)
            np.testing.assert_allclose(r_wo @ [0, 0, 1], [0, 0, 1], atol=1e-12)
            np.testing.assert_allclose((r_to @ ZERO_ROTATION_TO.T) @ NORMAL_T, NORMAL_T, atol=1e-12)
            expected_normal = tcp_rotation_for_upright_object(0, yaw) @ NORMAL_T
            np.testing.assert_allclose(r_wt @ NORMAL_T, expected_normal, atol=1e-12)
            measured, residual = measure_normal_angle(r_to)
            np.testing.assert_allclose(measured, angle, atol=1e-10)
            assert residual < 1e-10
    # A rotation about the approach axis is detectable and rejected by the constraint.
    wrong = Rotation.from_euler('z', 10, degrees=True).as_matrix() @ ZERO_ROTATION_TO
    assert measure_normal_angle(wrong)[1] > 9.9


def test_angle_bins_replay_and_center_offset():
    center_o = np.array([0., 0., .025])
    edges = np.linspace(-15, 15, 6)
    for b in range(5):
        a = propose_normal_pose(80000, 2, b, 1, 0, edges, center_o)
        replay = propose_normal_pose(80000, 2, b, 1, 0, edges, center_o)
        for key in a: np.testing.assert_array_equal(a[key], replay[key])
        assert edges[b] <= a['angle_deg'] <= edges[b+1]
        np.testing.assert_allclose(a['rotation_to'] @ center_o + a['position_to_m'], a['center_t_m'], atol=1e-12)
        assert .2 <= a['depth_mm'] <= 1.
        assert np.max(np.abs(a['offset_xy_mm'])) <= 2.


def test_ik_accepts_float32_simulator_joint_input():
    chain = TcpKinematics([((np.eye(4), np.eye(4), np.array([0., 1., 0.]), i), False)
                           for i in range(7)], np.eye(4))
    target = Rotation.from_euler('y', 15, degrees=True).as_matrix()
    q, diagnostic = chain.solve(np.zeros(3), target, np.zeros(7, dtype=np.float32),
                                np.tile([-1., 1.], (7, 1)))
    assert diagnostic['fk_rotation_error_deg'] < .001
    np.testing.assert_allclose(chain.forward(q)[:3, :3], target, atol=1e-5)


def test_single_angle_output_and_gradients_use_both_images_and_opening():
    torch.set_num_threads(2); torch.manual_seed(0)
    net = NormalAngleNet()
    left = torch.rand(2, 3, 120, 160, requires_grad=True)
    right = torch.rand(2, 3, 120, 160, requires_grad=True)
    opening = torch.tensor([[.002], [.010]], requires_grad=True)
    prediction = net(left, right, opening)
    assert prediction.shape == (2,) and prediction.abs().max() <= 15
    ((prediction-torch.tensor([-8., 11.]))**2).mean().backward()
    for x in (left, right, opening):
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert net.fusion[-1].out_features == 1


def test_angle_splits_and_fresh_data_guard_against_leakage(tmp_path):
    def write(path, uid, seed_offset):
        meta = {'tactile_calibration': {'id':'test-calibration'}, 'pose_constraint':'upright',
                'angle_convention':'single_normal', 'zero_rotation_to':ZERO_ROTATION_TO.tolist(),
                'object_yaw_deg':180., 'angle_range_deg':[-15.,15.]}
        with h5py.File(path, 'w') as f:
            f.attrs.update(schema=SCHEMA, status='complete', dataset_uuid=uid, metadata_json=json.dumps(meta))
            f['quality/sample_valid'] = np.ones(50,dtype=bool)
            f['ids/contact_pose_id'] = np.arange(50)
            f['ids/object_id'] = np.zeros(50,dtype=int)
            f['ids/angle_bin'] = np.repeat(np.arange(5),10)
            f['random/seed'] = np.arange(50)+seed_offset
    original, fresh = tmp_path/'original.h5', tmp_path/'fresh.h5'
    write(original,'original',0)
    m = split_manifest([original], seed=0)
    groups = [set(r['group'] for r in m['splits'][name]) for name in ('train','validation','test')]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert set.union(*groups) == {f'original:{i}' for i in range(50)}
    for name in m['splits']:
        assert set(r['angle_bin'] for r in m['splits'][name]) == set(range(5))
    with pytest.raises(ValueError,match='UUID'): fresh_manifest([original],m)
    write(fresh,'fresh',0)
    with pytest.raises(ValueError,match='seeds overlap'): fresh_manifest([fresh],m)
    write(fresh,'fresh',100)
    assert len(fresh_manifest([fresh],m)['splits']['external']) == 50
