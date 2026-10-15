import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tactile_pose.tcp_sampling import approach_directions, object_to_tcp_rotation, propose_pose, relative_pose_numpy


def test_approach_roll_and_perpendicular_translation():
    center_o = np.array([0., 0., .025])
    for i, d in enumerate(approach_directions(20)):
        a = propose_pose(42, 2, i, 0, 0, d, center_o)
        r = a["rotation_to"]
        np.testing.assert_allclose(r @ d, [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.)
        np.testing.assert_allclose(r @ center_o + a["position_to_m"], a["center_t_m"], atol=1e-12)
        # Translate the center in the approach-perpendicular plane in object coordinates.
        assert np.dot(r.T @ a["center_t_m"], d) == pytest.approx(0., abs=1e-12)
        assert .2 <= a["depth_mm"] <= 1.
    np.testing.assert_allclose(object_to_tcp_rotation([0, 0, 1], 90) @ [1, 0, 0], [0, -1, 0], atol=1e-12)


def test_streams_are_reproducible_and_distinct_across_objects_and_retries():
    args = (100, 2, 0, 0, 0, [0, 0, 1], [0, 0, .025])
    a, b = propose_pose(*args), propose_pose(*args)
    for key in a: np.testing.assert_array_equal(a[key], b[key])
    seeds = {propose_pose(100, obj, 0, 0, retry, [0, 0, 1], [0, 0, .025])["seed"]
             for obj in range(6) for retry in range(20)}
    assert len(seeds) == 120


def test_tcp_target_is_actual_pose_and_reconstructs_world_point():
    rng = np.random.default_rng(42)
    rt, ro = Rotation.random(2, random_state=rng)
    pt, po, point = rng.normal(size=(3, 3))
    p, r, q = relative_pose_numpy(pt, rt.as_quat()[[3, 0, 1, 2]], po, ro.as_quat()[[3, 0, 1, 2]])
    np.testing.assert_allclose(rt.apply(r @ point + p) + pt, ro.apply(point) + po, atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(), r, atol=1e-12)
    with pytest.raises(ValueError): object_to_tcp_rotation([0, 0, 0], 0)
