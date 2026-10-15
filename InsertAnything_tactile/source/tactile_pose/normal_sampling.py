"""Single in-hand rotation about the grasp normal, with a world-upright peg."""
import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = "tactile_tcp_normal_v4"
CONSTRAINT_BIT = 128
NORMAL_T = np.array([0., 1., 0.])
# Upstream insertion grasp: TCP +Z points down, peg +Z points up.
ZERO_ROTATION_TO = Rotation.from_euler("y", 180., degrees=True).as_matrix()


def world_object_rotation(yaw_deg=180.):
    return Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()


def normal_rotation_to(angle_deg):
    return Rotation.from_euler("y", angle_deg, degrees=True).as_matrix() @ ZERO_ROTATION_TO


def tcp_rotation_for_upright_object(angle_deg, object_yaw_deg=180.):
    return world_object_rotation(object_yaw_deg) @ normal_rotation_to(angle_deg).T


def measure_normal_angle(rotation_to):
    """Signed object-in-TCP deviation about +TCP Y from the insertion grasp."""
    delta = np.asarray(rotation_to) @ ZERO_ROTATION_TO.T
    angle = np.arctan2(delta[..., 0, 2], delta[..., 0, 0])
    ideal = Rotation.from_euler("y", angle).as_matrix()
    residual = ideal.swapaxes(-1, -2) @ delta
    error = np.rad2deg(Rotation.from_matrix(residual).magnitude())
    return np.rad2deg(angle), error


def propose_normal_pose(root_seed, object_id, angle_bin, slot, retry, angle_edges,
                        bbox_center_o, offset_mm=(2., 2.), depth_range_mm=(.2, 1.), center_z_mm=0.):
    sequence = np.random.SeedSequence([4, root_seed, object_id, angle_bin, slot, retry])
    seed = int(sequence.generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63 - 1))
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(angle_edges[angle_bin], angle_edges[angle_bin + 1]))
    offset = rng.uniform(-np.asarray(offset_mm), np.asarray(offset_mm))
    center_t = np.r_[offset, center_z_mm] / 1000.
    rotation_to = normal_rotation_to(angle)
    return {"seed": seed, "angle_deg": angle, "offset_xy_mm": offset, "center_t_m": center_t,
            "rotation_to": rotation_to, "position_to_m": center_t - rotation_to @ bbox_center_o,
            "depth_mm": float(rng.uniform(*depth_range_mm))}
