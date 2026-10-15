"""Isaac-only static contact rendering; import after AppLauncher.

The joint configuration is written directly and physics is not advanced between
samples. Geometric overlap is the optical input, not deformable-gel mechanics.
"""
import json
import time
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pxr import Gf, Usd, UsdGeom
from .tcp_sampling import relative_pose_numpy, STATUS_BITS


def cpu(x):
    return x.detach().cpu().numpy().copy()


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]]).as_matrix()


def mesh_points(stage, path):
    """Mesh points in the held asset's root frame, with authored transforms."""
    root = stage.GetPrimAtPath(path)
    cache = UsdGeom.XformCache()
    inverse_root = cache.GetLocalToWorldTransform(root).GetInverse()
    points = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            transform = cache.GetLocalToWorldTransform(prim) * inverse_root
            points.extend(transform.Transform(Gf.Vec3d(*map(float, v)))
                          for v in UsdGeom.Mesh(prim).GetPointsAttr().Get())
    if not points:
        raise ValueError(f"No mesh points in {path}")
    return np.asarray(points)


class StaticContactSampler:
    def __init__(self, env, metadata):
        if env.num_envs != 1:
            raise ValueError("Static TCP collector currently uses one environment per process")
        self.env, self.metadata = env, metadata
        self.robot = env._robot
        self.tcp_index = self.robot.body_names.index("panda_fingertip_centered")
        self.finger_indices = [self.robot.joint_names.index(f"panda_finger_joint{i}") for i in (1, 2)]
        self.case_indices = [self.robot.body_names.index(f"gelsight_mini_case_{s}") for s in ("left", "right")]
        self.points_o = mesh_points(env.sim.stage, "/World/envs/env_0/HeldAsset")
        self.center_o = (self.points_o.min(0) + self.points_o.max(0)) / 2
        self.joints = self.robot.data.joint_pos.clone()
        self.limits = cpu(self.robot.data.joint_pos_limits)[0, self.finger_indices]
        self.p_wt = cpu(self.robot.data.body_pos_w)[0, self.tcp_index]
        self.q_wt = cpu(self.robot.data.body_quat_w)[0, self.tcp_index]
        self.r_wt = rotation(self.q_wt)
        self.camera_in_case = []
        cache = UsdGeom.XformCache()
        for side in ("left", "right"):
            case = env.sim.stage.GetPrimAtPath(f"/World/envs/env_0/Robot/gelsight_mini_case_{side}")
            cam = env.sim.stage.GetPrimAtPath(str(case.GetPath()) + "/Camera")
            self.camera_in_case.append(np.array(cache.GetLocalToWorldTransform(cam) *
                                                 cache.GetLocalToWorldTransform(case).GetInverse()).T)
        # Move the hole away for this collection instance only. Keep arm pose
        # and all upstream task configurations unchanged.
        state = env._fixed_asset.data.root_state_w.clone()
        state[:, 0] += 2.
        state[:, 7:] = 0
        env._fixed_asset.write_root_state_to_sim(state)
        self.set_fingers(np.array([.020, .020]))
        env.sim.forward()
        self.plane_y_at_zero = []
        self.gel_centers_t = []
        for i, side in enumerate(("left", "right")):
            p, r = self.camera_world(i)
            zero_m = env.tactile_calibration["zero_reference_mm"][side] / 1000.
            front_t = self.r_wt.T @ (p + r @ np.array([0, 0, -zero_m]) - self.p_wt)
            normal_t = self.r_wt.T @ r @ np.array([0, 0, -1.])
            np.testing.assert_allclose(normal_t, [0, -1 if i == 0 else 1, 0], atol=1e-4)
            self.plane_y_at_zero.append(front_t[1] - (.020 if i == 0 else -.020))
            self.gel_centers_t.append(front_t.tolist())
        metadata.update(object_bbox_o_m=[self.points_o.min(0).tolist(), self.points_o.max(0).tolist()],
                        object_bbox_center_o_m=self.center_o.tolist(),
                        finger_joint_limits_m=self.limits.tolist(),
                        gel_front_y_at_zero_joint_m=self.plane_y_at_zero,
                        gel_front_centers_at_20mm_joints_t_m=self.gel_centers_t,
                        robot_joint_names=self.robot.joint_names,
                        tactile_calibration=env.tactile_calibration,
                        initialization_sim_time_s=float(env.sim.current_time))
        self.render_count = 0

    def camera_world(self, i):
        p = cpu(self.robot.data.body_pos_w)[0, self.case_indices[i]]
        r = rotation(cpu(self.robot.data.body_quat_w)[0, self.case_indices[i]])
        local = self.camera_in_case[i]
        return p + r @ local[:3, 3], r @ local[:3, :3]

    def set_fingers(self, q):
        self.joints[:, self.finger_indices] = torch.as_tensor(q, dtype=torch.float32, device=self.env.device)
        self.robot.write_joint_state_to_sim(self.joints, torch.zeros_like(self.joints))

    def render(self):
        self.render_count += 1
        return self.env.refresh_tactile()

    def bounds_feasible(self, proposal):
        # A necessary (not sufficient) condition. If even the entire mesh cannot
        # reach the requested overlap at the lower joint limit, skip rendering.
        points_t = self.points_o @ proposal["rotation_to"].T + proposal["position_to_m"]
        required = np.array([points_t[:, 1].max() - self.plane_y_at_zero[0],
                             self.plane_y_at_zero[1] - points_t[:, 1].min()]) - proposal["depth_mm"] / 1000
        return bool(np.all(required >= self.limits[:, 0]))

    def capture(self, proposal, identity, *, sampling_fields=None):
        env, meta = self.env, self.metadata
        state = env._held_asset.data.root_state_w.clone()
        state[:, :3] = torch.as_tensor(self.p_wt + self.r_wt @ proposal["position_to_m"], device=env.device)
        q_wo = Rotation.from_matrix(self.r_wt @ proposal["rotation_to"]).as_quat()[[3, 0, 1, 2]]
        state[:, 3:7] = torch.as_tensor(q_wo, device=env.device)
        state[:, 7:] = 0
        env._held_asset.write_root_state_to_sim(state)
        points_t = self.points_o @ proposal["rotation_to"].T + proposal["position_to_m"]
        d = proposal["depth_mm"]
        q = np.array([points_t[:, 1].max() - self.plane_y_at_zero[0] - d / 1000,
                      self.plane_y_at_zero[1] - points_t[:, 1].min() - d / 1000])
        q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        for iteration in range(meta["max_depth_iterations"]):
            self.set_fingers(q)
            tactile = self.render()
            # Signed separation remains useful before contact. A missed ray is
            # far clip; bound each correction so the object cannot jump through
            # the near clipping plane in one update.
            signed = np.array([env.tactile_calibration["zero_reference_mm"][s] -
                               float(tactile[s]["camera_depth_mm"].min()) for s in ("left", "right")])
            if np.all(np.abs(signed - d) <= meta["depth_tolerance_mm"]):
                break
            adjusted = np.clip(q + np.clip((signed - d) / 1000., -.0015, .0015),
                               self.limits[:, 0], self.limits[:, 1])
            if np.max(np.abs(adjusted - q)) < 1e-7:
                break
            if iteration + 1 < meta["max_depth_iterations"]:
                q = adjusted
        # All labels are read AFTER the final pose write and synchronized render.
        pt = cpu(self.robot.data.body_pos_w)[0, self.tcp_index]
        qt = cpu(self.robot.data.body_quat_w)[0, self.tcp_index]
        po = cpu(env._held_asset.data.root_pos_w)[0]
        qo = cpu(env._held_asset.data.root_quat_w)[0]
        p_to, r_to, q_to = relative_pose_numpy(pt, qt, po, qo)
        actual_q = cpu(self.robot.data.joint_pos)[0, self.finger_indices]
        if sampling_fields is None:
            sampling_fields = {
                "sampling/approach_direction_o": np.asarray(meta["approach_directions_o"])[int(identity["ids/direction_id"])],
                "sampling/roll_about_approach_deg": np.float32(proposal["roll_deg"]),
            }
        row = {
            **identity, **sampling_fields, "random/seed": np.int64(proposal["seed"]),
            "sampling/offset_xy_mm": proposal["offset_xy_mm"].astype(np.float32),
            "sampling/requested_center_t_m": proposal["center_t_m"].astype(np.float32),
            "sampling/requested_rotation_to": proposal["rotation_to"].astype(np.float32),
            "sampling/requested_depth_mm": np.float32(d),
            "sampling/depth_iterations": np.int32(iteration + 1),
            "time/unix_ns": np.int64(time.time_ns()),
            "time/sim_time_s": np.float64(env.sim.current_time),
            "time/physics_step": np.int64(env.sim.current_time_step_index),
            "time/render_refresh_index": np.int64(self.render_count),
            "tcp/position_w_m": pt, "tcp/quaternion_wxyz": qt,
            "object/position_w_m": po, "object/quaternion_wxyz": qo,
            "target/position_to_m": p_to.astype(np.float32),
            "target/rotation_to": r_to.astype(np.float32),
            "target/quaternion_to_wxyz": q_to.astype(np.float32),
            "object/bbox_center_t_m": (r_to @ self.center_o + p_to).astype(np.float32),
            "gripper/finger_joint_position_m": actual_q,
            "gripper/opening_joint_sum_m": np.array([actual_q.sum()], dtype=np.float32),
            "gripper/gel_front_gap_m": np.float32(actual_q.sum() + self.plane_y_at_zero[0] - self.plane_y_at_zero[1]),
            "robot/joint_position": cpu(self.robot.data.joint_pos)[0],
            "robot/joint_velocity": cpu(self.robot.data.joint_vel)[0],
        }
        flags = 0
        for i, side in enumerate(("left", "right")):
            for key in ("rgb", "press_depth_mm", "camera_depth_mm"):
                row[f"tactile/{side}/{key}"] = cpu(tactile[side][key])[0]
            press = row[f"tactile/{side}/press_depth_mm"]
            depth = row[f"tactile/{side}/camera_depth_mm"]
            pixels = int((press > meta["contact_depth_threshold_mm"]).sum())
            maximum = float(press.max())
            row[f"contact/{side}_pixels"] = np.int32(pixels)
            row[f"contact/{side}_max_press_mm"] = np.float32(maximum)
            row[f"contact/{side}_depth_error_mm"] = np.float32(maximum - d)
            valid = pixels >= meta["contact_min_pixels"]
            row[f"contact/{side}_valid"] = np.bool_(valid)
            if not valid: flags |= 1 << i
            if abs(maximum - d) > meta["depth_tolerance_mm"]: flags |= 8
            near = env.cfg.tactile.clipping_range[0] * 1000
            saturated = int(((depth <= near + .02) | (press >= meta["tactile_calibration"]["gel_height_mm"] - .02)).sum())
            row[f"contact/{side}_clipped_pixels"] = np.int32(saturated)
            if saturated: flags |= 32
            cam_p, cam_r = self.camera_world(i)
            row[f"sensor/{side}_camera_position_w_m"] = cam_p.astype(np.float32)
            row[f"sensor/{side}_camera_quaternion_wxyz"] = Rotation.from_matrix(cam_r).as_quat()[[3, 0, 1, 2]].astype(np.float32)
        if not all(np.isfinite(value).all() for value in row.values()): flags |= 4
        if np.any(actual_q < self.limits[:, 0] - 1e-6) or np.any(actual_q > self.limits[:, 1] + 1e-6): flags |= 16
        if np.linalg.norm(p_to - proposal["position_to_m"]) > 1e-5 or np.max(np.abs(r_to - proposal["rotation_to"])) > 1e-4:
            flags |= 64
        row["quality/status_code"] = np.uint8(flags)
        row["quality/sample_valid"] = np.bool_(flags == 0)
        row["quality/joint_at_lower_limit"] = actual_q <= self.limits[:, 0] + 1e-6
        row["contact/both_valid"] = np.bool_(row["contact/left_valid"] and row["contact/right_valid"])
        return row


class ContactWriter:
    def __init__(self, path, metadata):
        self.file = h5py.File(path, "x")
        self.meta = metadata
        self.count = self.valid_count = 0
        self.file.attrs.update(schema=metadata["schema"], dataset_uuid=metadata["dataset_uuid"],
                               status="collecting", metadata_json=json.dumps(metadata))

    def append(self, row):
        for key, value in row.items():
            value = np.asarray(value)
            if key not in self.file:
                self.file.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                    dtype=value.dtype, chunks=(1 if value.ndim > 1 else 128, *value.shape), compression="lzf", shuffle=True)
            ds = self.file[key]
            ds.resize(self.count + 1, axis=0)
            ds[self.count] = value
        self.count += 1
        self.valid_count += int(row["quality/sample_valid"])

    def finish(self, status):
        flags = self.file["quality/status_code"][:]
        valid = flags == 0
        errors = np.stack([self.file[f"contact/{s}_depth_error_mm"][:] for s in ("left", "right")], axis=1)
        ranges = {s: ([float(self.file[f"contact/{s}_max_press_mm"][:][valid].min()),
                       float(self.file[f"contact/{s}_max_press_mm"][:][valid].max())] if valid.any() else [])
                  for s in ("left", "right")}
        self.meta.update(status=status, attempts=self.count, valid_samples=self.valid_count,
            rejected_samples=self.count-self.valid_count,
            rejection_reasons={name: int(((flags & bit) != 0).sum()) for bit, name in STATUS_BITS.items()},
            valid_depth_range_mm=ranges,
            valid_max_abs_depth_error_mm=float(np.abs(errors[valid]).max()) if valid.any() else None)
        self.file.attrs.update(status=status, metadata_json=json.dumps(self.meta),
                               samples=self.count, valid_samples=self.valid_count)
        self.file.flush()

    def close(self):
        self.file.close()
