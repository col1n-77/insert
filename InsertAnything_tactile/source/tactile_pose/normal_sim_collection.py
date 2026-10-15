"""World-upright object; actual Panda wrist tilt about the grasp normal."""
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .tcp_sim_collection import StaticContactSampler, ContactWriter, cpu, rotation
from .tcp_kinematics import TcpKinematics, transform
from .normal_sampling import (NORMAL_T, ZERO_ROTATION_TO, CONSTRAINT_BIT,
    world_object_rotation, tcp_rotation_for_upright_object, measure_normal_angle)


class NormalContactSampler(StaticContactSampler):
    def __init__(self, env, metadata):
        super().__init__(env, metadata)
        self.anchor_w = self.p_wt.copy()
        base = self.robot.body_names.index("panda_link0")
        self.base_index = base
        self.base_transform = transform(cpu(self.robot.data.body_pos_w)[0, base],
                                        cpu(self.robot.data.body_quat_w)[0, base])
        self.chain = TcpKinematics.from_usd(env.sim.stage, "/World/envs/env_0/Robot",
                                           self.robot.joint_names, self.base_transform)
        joints = cpu(self.robot.data.joint_pos)[0]
        actual = self.chain.forward(joints)
        pe = float(np.linalg.norm(actual[:3, 3] - self.p_wt))
        re = float(np.rad2deg(Rotation.from_matrix(actual[:3, :3].T @ self.r_wt).magnitude()))
        if pe > 1e-5 or re > .005:
            raise RuntimeError(f"USD FK does not match PhysX: {pe} m, {re} deg")
        self.arm_limits = cpu(self.robot.data.joint_pos_limits)[0]
        self.object_rotation_w = world_object_rotation(metadata["object_yaw_deg"])
        metadata.update(kinematics_initial_check={"position_error_m": pe, "rotation_error_deg": re},
                        tcp_anchor_w_m=self.anchor_w.tolist(),
                        robot_joint_limits=self.arm_limits.tolist(),
                        zero_rotation_to=ZERO_ROTATION_TO.tolist(),
                        fixed_object_rotation_w=self.object_rotation_w.tolist(),
                        robot_base_transform_w=self.base_transform.tolist())

    def move_hand(self, angle_deg):
        target = tcp_rotation_for_upright_object(angle_deg, self.metadata["object_yaw_deg"])
        joints, diagnostic = self.chain.solve(self.anchor_w, target, cpu(self.joints)[0], self.arm_limits)
        self.joints[:] = torch.as_tensor(joints, device=self.env.device, dtype=self.joints.dtype)
        self.robot.write_joint_state_to_sim(self.joints, torch.zeros_like(self.joints))
        self.env.sim.forward()
        self.p_wt = cpu(self.robot.data.body_pos_w)[0, self.tcp_index]
        self.q_wt = cpu(self.robot.data.body_quat_w)[0, self.tcp_index]
        self.r_wt = rotation(self.q_wt)
        diagnostic["actual_position_error_m"] = float(np.linalg.norm(self.p_wt - self.anchor_w))
        diagnostic["actual_rotation_error_deg"] = float(np.rad2deg(Rotation.from_matrix(self.r_wt.T @ target).magnitude()))
        if diagnostic["actual_position_error_m"] > 1e-5 or diagnostic["actual_rotation_error_deg"] > .005:
            raise RuntimeError(f"PhysX pose after IK write failed verification: {diagnostic}")
        return diagnostic

    def capture(self, proposal, identity):
        diagnostic = self.move_hand(proposal["angle_deg"])
        # Recompute from the ACTUAL wrist pose to keep object +Z exactly world
        # +Z, even with residual float32 FK/IK error. Only labels are measured.
        applied = dict(proposal)
        applied["rotation_to"] = self.r_wt.T @ self.object_rotation_w
        applied["position_to_m"] = proposal["center_t_m"] - applied["rotation_to"] @ self.center_o
        row = super().capture(applied, identity, sampling_fields={
            "sampling/requested_angle_deg": np.float32(proposal["angle_deg"]),
            "sampling/rotation_axis_t": NORMAL_T.astype(np.float32),
            "sampling/nominal_requested_rotation_to": proposal["rotation_to"].astype(np.float32),
            "sampling/ik_position_error_m": np.float32(diagnostic["actual_position_error_m"]),
            "sampling/ik_rotation_error_deg": np.float32(diagnostic["actual_rotation_error_deg"]),
            "sampling/ik_nfev": np.int32(diagnostic["nfev"]),
        })
        measured, residual = measure_normal_angle(row["target/rotation_to"])
        axis = rotation(row["object/quaternion_wxyz"]) @ np.array([0., 0., 1.])
        vertical_error = float(np.rad2deg(np.arctan2(np.linalg.norm(axis[:2]), axis[2])))
        row["target/in_hand_angle_deg"] = np.float32(measured)
        row["quality/off_axis_rotation_deg"] = np.float32(residual)
        row["quality/object_vertical_error_deg"] = np.float32(vertical_error)
        row["object/axis_w"] = axis.astype(np.float32)
        base = transform(cpu(self.robot.data.body_pos_w)[0, self.base_index],
                         cpu(self.robot.data.body_quat_w)[0, self.base_index])
        row["quality/robot_base_max_change"] = np.float32(np.max(np.abs(base - self.base_transform)))
        constraint_valid = (vertical_error < .005 and residual < .005 and
                            abs(measured - proposal["angle_deg"]) < .005 and
                            row["quality/robot_base_max_change"] < 1e-6)
        if not constraint_valid: row["quality/status_code"] |= np.uint8(CONSTRAINT_BIT)
        row["quality/sample_valid"] = np.bool_(row["quality/status_code"] == 0)
        return row


class NormalWriter(ContactWriter):
    def finish(self, status):
        super().finish(status)
        flags = self.file["quality/status_code"][:]
        self.meta["rejection_reasons"]["upright_or_single_axis_constraint_invalid"] = int(((flags & CONSTRAINT_BIT) != 0).sum())
        valid = flags == 0
        for key in ("off_axis_rotation_deg", "object_vertical_error_deg", "robot_base_max_change"):
            self.meta["valid_max_" + key] = float(self.file["quality/" + key][:][valid].max()) if valid.any() else None
        angles = self.file["target/in_hand_angle_deg"][:][valid]
        self.meta["valid_angle_range_deg"] = [float(angles.min()), float(angles.max())] if valid.any() else []
        self.file.attrs["metadata_json"] = json.dumps(self.meta)
        self.file.flush()
