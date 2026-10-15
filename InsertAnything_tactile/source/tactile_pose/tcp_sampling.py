"""Independent contact-pose proposals. Pure NumPy; no simulator imports."""
import numpy as np
from scipy.spatial.transform import Rotation

TASK_SUFFIXES = ("CircleHole-I", "SquareHole-II", "LHole-III", "TriangleHole-IV",
                 "HexagonHole-III", "HexagonHole-IV")
SCHEMA = "tactile_tcp_pose_v3"
STATUS_BITS = {1: "left_contact_invalid", 2: "right_contact_invalid", 4: "nonfinite",
               8: "indentation_target_unreachable", 16: "joint_limit_invalid",
               32: "depth_clipped_or_saturated", 64: "pose_write_mismatch"}


def approach_directions(tilt_deg=20.0):
    """TCP +Z expressed in object coordinates: axial, +/-X tilt, +/-Y tilt."""
    if not np.isfinite(tilt_deg) or not 0 < tilt_deg < 90:
        raise ValueError("tilt_deg must lie strictly between 0 and 90 degrees")
    s, c = np.sin(np.deg2rad(tilt_deg)), np.cos(np.deg2rad(tilt_deg))
    return np.array([[0, 0, 1], [s, 0, c], [-s, 0, c], [0, s, c], [0, -s, c]])


def object_to_tcp_rotation(direction_o, roll_deg):
    """R_TO: object column vectors -> TCP. Roll is TCP motion about its +Z."""
    z = np.asarray(direction_o, dtype=np.float64)
    if z.shape != (3,) or not np.isfinite(z).all() or np.linalg.norm(z) < 1e-8:
        raise ValueError("An approach direction must be a finite nonzero 3-vector")
    z = z / np.linalg.norm(z)
    reference = np.array([1., 0., 0.]) if abs(z[0]) < .9 else np.array([0., 1., 0.])
    x = reference - np.dot(reference, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return (np.column_stack((x, y, z)) @ Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()).T


def propose_pose(root_seed, object_id, direction_id, slot, attempt, direction_o,
                 bbox_center_o, offset_mm=(2., 2.), roll_range_deg=(-180., 180.),
                 depth_range_mm=(.2, 1.), center_z_mm=0.):
    # Independent streams and explicit provenance, including rejected attempts.
    sequence = np.random.SeedSequence([root_seed, object_id, direction_id, slot, attempt])
    seed = int(sequence.generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63 - 1))
    rng = np.random.default_rng(seed)
    roll = rng.uniform(*roll_range_deg)
    offset = rng.uniform(-np.asarray(offset_mm), np.asarray(offset_mm))
    center_t = np.r_[offset, center_z_mm] / 1000.
    rotation_to = object_to_tcp_rotation(direction_o, roll)
    return {"seed": seed, "roll_deg": roll, "offset_xy_mm": offset,
            "center_t_m": center_t, "rotation_to": rotation_to,
            "position_to_m": center_t - rotation_to @ np.asarray(bbox_center_o),
            "depth_mm": float(rng.uniform(*depth_range_mm))}


def relative_pose_numpy(p_wt, q_wt, p_wo, q_wo):
    """Actual rigid-body poses, quaternion wxyz; no requested pose used as label."""
    rt = Rotation.from_quat(np.asarray(q_wt)[..., [1, 2, 3, 0]])
    ro = Rotation.from_quat(np.asarray(q_wo)[..., [1, 2, 3, 0]])
    relative = rt.inv() * ro
    return rt.inv().apply(np.asarray(p_wo) - p_wt), relative.as_matrix(), relative.as_quat()[..., [3, 0, 1, 2]]
