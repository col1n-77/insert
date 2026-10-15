"""Isaac-only data collection; import after AppLauncher."""
import json
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import isaacsim.core.utils.torch as torch_utils

from insertanything_tactile.recording import RecordingEnv
from .rotations import relative_pose


class PoseCollectionEnv(RecordingEnv):
    def __init__(self, *args, **kwargs):
        self.grasp_perturbation_wxyz = None
        super().__init__(*args, **kwargs)

    def get_handheld_asset_relative_pose(self):
        position, quaternion = super().get_handheld_asset_relative_pose()
        if self.grasp_perturbation_wxyz is not None:
            quaternion = torch_utils.quat_mul(quaternion, self.grasp_perturbation_wxyz)
        # Upstream inverts this transform, places the object, and physically
        # closes the fingers. Requested rotations are NOT ground-truth labels.
        return position, quaternion


class PoseWriter:
    def __init__(self, path, env, metadata):
        self.file = h5py.File(path, "w")
        self.file.attrs["schema"] = metadata["schema"]
        self.file.attrs["status"] = "collecting"
        self.file.attrs["metadata_json"] = json.dumps(metadata)
        self.file.attrs["dataset_uuid"] = metadata["dataset_uuid"]
        self.count = 0
        self.valid_count = 0
        self.gripper_index = env._robot.body_names.index("panda_hand")
        self.finger_indices = [env._robot.joint_names.index(name) for name in
                               ("panda_finger_joint1", "panda_finger_joint2")]
        self.side_indices = [env._robot.body_names.index(f"gelsight_mini_case_{side}")
                             for side in ("left", "right")]
        self.metadata = metadata
        self.grasp_ids = None
        self.seeds = None
        self.reset_seed = None
        self.requested_angles = None

    def begin_grasps(self, grasp_ids, seeds, reset_seed, angles):
        self.grasp_ids = np.asarray(grasp_ids, dtype=np.int64)
        self.seeds = np.asarray(seeds, dtype=np.int64)
        self.reset_seed = reset_seed
        self.requested_angles = np.asarray(angles, dtype=np.float32)

    def capture(self, env, reward):
        if self.grasp_ids is None:
            raise RuntimeError("A grasp must be identified before recording")
        n = env.num_envs
        cpu = lambda value: value.detach().cpu().numpy().copy()
        # These are rigid-body link poses, not noisy policy observations or targets.
        p_wg = env._robot.data.body_pos_w[:, self.gripper_index]
        q_wg = env._robot.data.body_quat_w[:, self.gripper_index]
        p_wo = env._held_asset.data.root_pos_w
        q_wo = env._held_asset.data.root_quat_w
        p_go, r_go = relative_pose(p_wg, q_wg, p_wo, q_wo)
        r_np = cpu(r_go)
        q_go = Rotation.from_matrix(r_np).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
        q_go *= np.where(q_go[:, :1] < 0, -1, 1)
        finger_position = cpu(env._robot.data.joint_pos[:, self.finger_indices])
        steps = cpu(env.episode_length_buf).astype(np.int32)
        row = {
            "ids/env_id": np.arange(n, dtype=np.int32),
            "ids/grasp_id": self.grasp_ids,
            "ids/object_id": np.zeros(n, dtype=np.int32),
            "random/seed": self.seeds,
            "random/reset_seed": np.full(n, self.reset_seed, dtype=np.int64),
            "random/requested_euler_xyz_deg": self.requested_angles,
            "time/grasp_step": steps,
            "time/grasp_time_s": steps.astype(np.float64) * env.step_dt,
            "time/policy_physics_step": np.full(n, env._sim_step_counter, dtype=np.int64),
            "time/sim_time_s": np.full(n, float(env.sim.current_time), dtype=np.float64),
            "gripper/finger_joint_position_m": finger_position,
            "gripper/opening_joint_sum_m": finger_position.sum(-1, keepdims=True),
            "gripper/position_w_m": cpu(p_wg),
            "gripper/quaternion_wxyz": cpu(q_wg),
            "object/position_w_m": cpu(p_wo),
            "object/quaternion_wxyz": cpu(q_wo),
            "target/position_go_m": cpu(p_go),
            "target/rotation_go": r_np,
            "target/quaternion_go_wxyz": q_go,
            "robot/joint_position": cpu(env._robot.data.joint_pos),
            "robot/joint_velocity": cpu(env._robot.data.joint_vel),
            "sensor/position_w_m": cpu(env._robot.data.body_pos_w[:, self.side_indices]),
            "sensor/quaternion_wxyz": cpu(env._robot.data.body_quat_w[:, self.side_indices]),
        }
        tactile = env.refresh_tactile()
        flags = np.zeros(n, dtype=np.uint8)
        for side, bit in (("left", 1), ("right", 2)):
            rgb = cpu(tactile[side]["rgb"])
            press = cpu(tactile[side]["press_depth_mm"])
            row[f"tactile/{side}/rgb"] = rgb
            row[f"tactile/{side}/press_depth_mm"] = press
            row[f"tactile/{side}/camera_depth_mm"] = cpu(tactile[side]["camera_depth_mm"])
            pixels = (press > self.metadata["contact_depth_threshold_mm"]).sum((1, 2)).astype(np.int32)
            valid = pixels >= self.metadata["contact_min_pixels"]
            row[f"contact/{side}_pixels"] = pixels
            row[f"contact/{side}_max_press_mm"] = press.max((1, 2))
            row[f"contact/{side}_valid"] = valid
            flags[~valid] |= bit
        finite = np.ones(n, dtype=np.bool_)
        for value in row.values():
            finite &= np.isfinite(value.reshape(n, -1)).all(-1)
        flags[~finite] |= 4
        pose_valid = (np.linalg.norm(row["target/position_go_m"], axis=-1) < 0.25)
        opening = row["gripper/opening_joint_sum_m"][:, 0]
        pose_valid &= (opening > 0) & (opening < 0.081)
        flags[~pose_valid] |= 8
        row["contact/both_valid"] = row["contact/left_valid"] & row["contact/right_valid"]
        row["quality/status_code"] = flags
        row["quality/sample_valid"] = flags == 0
        row["quality/terminated"] = cpu(env.reset_terminated)
        row["quality/truncated"] = cpu(env.reset_time_outs)
        for key, value in row.items():
            if key not in self.file:
                self.file.create_dataset(key, shape=(0, *value.shape[1:]),
                    maxshape=(None, *value.shape[1:]), dtype=value.dtype,
                    chunks=(1 if value.ndim > 2 else 128, *value.shape[1:]),
                    compression="lzf", shuffle=True)
            ds = self.file[key]
            ds.resize(self.count + n, axis=0)
            ds[self.count:self.count+n] = value
        self.count += n
        self.valid_count += int((flags == 0).sum())

    def flush(self):
        self.file.flush()

    def finish(self):
        self.file.attrs["status"] = "complete"
        self.file.attrs["samples"] = self.count
        self.file.attrs["valid_samples"] = self.valid_count
        self.metadata.update(status="complete", samples=self.count, valid_samples=self.valid_count,
                             completed_grasps=len(np.unique(self.file["ids/grasp_id"][:])))
        self.file.attrs["metadata_json"] = json.dumps(self.metadata)
        self.file.flush()

    def close(self):
        self.file.close()
