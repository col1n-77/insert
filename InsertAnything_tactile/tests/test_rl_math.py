import math
import torch
from tactile_pose.rl_math import (rotation_vector_quat, quat_mul, quat_inverse, quat_apply,
    estimated_axis_w, local_z_w, axis_angle, gaussian_alignment, world_delta_target, measured_in_hand)


def q_axis(axis, degrees):
    r = torch.zeros((len(degrees), 3), dtype=torch.float64)
    r[:, axis] = torch.deg2rad(torch.tensor(degrees, dtype=torch.float64))
    return rotation_vector_quat(r)


def test_gripper_compensates_nonzero_in_hand_angle():
    theta = torch.deg2rad(torch.tensor([-15., -5., 0., 5., 15.], dtype=torch.float64))
    r_to = quat_mul(q_axis(1, [-15., -5., 0., 5., 15.]), q_axis(1, [180.]*5))
    object_w = q_axis(2, [180.]*5)
    tcp_w = quat_mul(object_w, quat_inverse(r_to))
    estimate = estimated_axis_w(tcp_w, theta)
    torch.testing.assert_close(estimate, local_z_w(object_w), atol=1e-10, rtol=0)
    torch.testing.assert_close(gaussian_alignment(axis_angle(estimate, local_z_w(object_w)), torch.ones(5).bool(), 5),
                               torch.ones(5, dtype=torch.float64))
    measured, residual = measured_in_hand(tcp_w, object_w)
    torch.testing.assert_close(measured, theta)
    assert residual.max() < 1e-10


def test_tilt_error_rotating_world_and_reversed_axis():
    tcp = q_axis(0, [180.])
    axis = estimated_axis_w(tcp, torch.tensor([math.radians(10.)], dtype=torch.float64))
    hole = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    alpha = axis_angle(axis, hole)
    torch.testing.assert_close(alpha, torch.tensor([math.radians(10.)], dtype=torch.float64))
    world = quat_mul(q_axis(0, [32.]), q_axis(2, [-25.]))
    torch.testing.assert_close(axis_angle(quat_apply(world, axis), quat_apply(world, hole)), alpha)
    assert axis_angle(-hole, hole).item() == math.pi
    assert gaussian_alignment(alpha, torch.tensor([False]), 5).item() == 0.
    assert gaussian_alignment(alpha, torch.tensor([True]), 5).item() < .02


def test_all_rotational_actions_survive_and_zero_is_finite():
    current = q_axis(0, [180.]*4)
    delta = torch.cat((torch.eye(3, dtype=torch.float64)*.02, torch.zeros((1,3),dtype=torch.float64)))
    target = world_delta_target(current, delta, .035)
    recovered = quat_mul(target, quat_inverse(current))
    torch.testing.assert_close(recovered, rotation_vector_quat(delta))
    torch.testing.assert_close(target[-1], current[-1])
    assert torch.isfinite(target).all()
    large = world_delta_target(current[:1], torch.ones((1,3),dtype=torch.float64)*10, .035)
    d = quat_mul(large, quat_inverse(current[:1]))
    assert 2*torch.atan2(d[:,1:].norm(), d[:,0].abs()).item() <= .035000001
