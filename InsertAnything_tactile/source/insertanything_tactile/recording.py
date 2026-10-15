"""Record one policy episode, sampling before DirectRLEnv's automatic reset."""
import csv
from contextlib import ExitStack
import json

import h5py
import imageio.v2 as imageio
import numpy as np

from .env import InsertAnythingTactileEnv


class RecordingEnv(InsertAnythingTactileEnv):
    def __init__(self, *args, **kwargs):
        self.recorder = None
        super().__init__(*args, **kwargs)

    def _get_rewards(self):
        reward = super()._get_rewards()
        # DirectRLEnv has updated state, done flags and metrics here, but has
        # not called _reset_idx yet. env.step() returns after that reset.
        if self.recorder is not None:
            self.recorder.capture(self, reward)
        return reward


def array(tensor):
    """Copy environment zero to CPU before the simulator reuses its buffers."""
    return tensor[0].detach().cpu().numpy().copy()


class EpisodeRecorder:
    def __init__(self, output, env, metadata):
        metadata["tactile_calibration"] = env.tactile_calibration
        self.output = output
        self.count = 0
        self.pending = None
        self.last = None
        self.total_reward = 0.0
        self.max_force = 0.0
        self.max_press = {side: 0.0 for side in env.tactile_sensors}
        self.stack = ExitStack()
        try:
            self.h5 = self.stack.enter_context(h5py.File(output / "trajectory.h5", "w"))
            self.h5.attrs["status"] = "in_progress"
            self.h5.attrs["metadata_json"] = json.dumps(metadata)
            self.h5.attrs["sample_alignment"] = (
                "Row i: observation/action before step i+1; state, reward, images and done "
                "after its physics, before auto-reset. No reset frames are included."
            )
            self.csv = self.stack.enter_context((output / "states.csv").open("w", newline=""))
            self.csv_writer = None
            self.videos = {}
            for name in ("scene", "left_tactile", "right_tactile"):
                writer = imageio.get_writer(
                    output / f"{name}.mp4", fps=1.0 / env.step_dt,
                    codec="libx264", quality=8, macro_block_size=1,
                    ffmpeg_params=["-threads", "2"],
                )
                self.stack.callback(writer.close)
                self.videos[name] = writer
        except BaseException:
            self.stack.close()
            raise

    def before_action(self, observations, actions, clip_obs):
        if self.pending is not None:
            raise RuntimeError("Previous action was not recorded")
        self.pending = {
            "policy/observation_before_action": array(observations["policy"]),
            "policy/input_after_clip": array(observations["policy"].clamp(-clip_obs, clip_obs)),
            "policy/critic_observation_before_action": array(observations["critic"]),
            "policy/action": array(actions),
        }

    def capture(self, env, reward):
        if self.pending is None:
            raise RuntimeError("Recording callback has no corresponding policy action")
        step = int(env.episode_length_buf[0])
        if step != self.count + 1:
            raise RuntimeError(f"Unexpected episode step {step} after {self.count} frames")

        sample = self.pending
        sample.update({
            "step": np.asarray(step, dtype=np.int32),
            "time_s": np.asarray(step * env.step_dt, dtype=np.float64),
            "policy/smoothed_action": array(env.actions),
            "robot/joint_position": array(env.joint_pos),
            "robot/joint_velocity": array(env.joint_vel),
            "robot/commanded_joint_effort": array(env.joint_torque),
            "robot/ee_position_env_m": array(env.fingertip_midpoint_pos),
            "robot/ee_quaternion_wxyz": array(env.fingertip_midpoint_quat),
            "robot/ee_linear_velocity_w_m_s": array(env.fingertip_midpoint_linvel),
            "robot/ee_angular_velocity_w_rad_s": array(env.fingertip_midpoint_angvel),
            "object/peg_position_env_m": array(env.held_pos),
            "object/peg_quaternion_wxyz": array(env.held_quat),
            "object/peg_linear_velocity_w_m_s": array(env._held_asset.data.root_lin_vel_w),
            "object/peg_angular_velocity_w_rad_s": array(env._held_asset.data.root_ang_vel_w),
            "object/hole_position_env_m": array(env.fixed_pos),
            "object/hole_quaternion_wxyz": array(env.fixed_quat),
            "force/contact_local_n": array(env.contact_force_local),
            "force/tare_local_n": array(env.contact_force_tare),
            "reward": array(reward),
            "terminated": array(env.reset_terminated),
            "truncated": array(env.reset_time_outs),
            "metrics/episode_succeeded": array(env.ep_succeeded).astype(np.bool_),
            "metrics/first_success_step": array(env.ep_success_times),
        })
        contact = env.scene.sensors.get("contact_sensor")
        if contact is not None:
            sample["force/net_normal_world_n"] = array(contact.data.net_forces_w.sum(dim=1))
        for name in ("curr_engaged", "curr_engaged_half", "curr_success"):
            sample[f"metrics/{name}"] = np.asarray(bool(env.extras[f"logs_rew_{name}"]))
        for name, value in env.extras.items():
            if name.startswith("logs_rew_"):
                sample[f"reward_terms/{name.removeprefix('logs_rew_')}"] = np.asarray(float(value))

        # Rendering only: no physics step and no extra noisy observation call.
        # This also avoids keeping a camera frame from before the final substep.
        tactile = env.refresh_tactile()
        scene = env.render()
        if scene is None or not scene.size or not np.any(scene):
            raise RuntimeError("Scene camera produced an empty frame")
        images = {}
        for side, values in tactile.items():
            for key, value in values.items():
                images[f"tactile/{side}/{key}"] = array(value)
            rgb = images[f"tactile/{side}/rgb"]
            width, height = env.cfg.tactile.resolution
            if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
                raise RuntimeError(f"Invalid {side} tactile RGB: {rgb.shape}, {rgb.dtype}")
            self.max_press[side] = max(self.max_press[side], float(images[f"tactile/{side}/press_depth_mm"].max()))

        for key, value in {**sample, **images}.items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"Non-finite {key} at step {step}")
            if key not in self.h5:
                self.h5.create_dataset(
                    key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                    dtype=value.dtype, chunks=(1 if value.ndim >= 2 else 128, *value.shape),
                    compression="lzf", shuffle=True,
                )
            dataset = self.h5[key]
            dataset.resize(self.count + 1, axis=0)
            dataset[self.count] = value

        row = {}
        for key, value in sample.items():
            if key in ("policy/observation_before_action", "policy/input_after_clip", "policy/critic_observation_before_action"):
                continue
            column = key.replace("/", "_")
            if value.ndim == 0:
                row[column] = value.item()
            else:
                row.update({f"{column}_{i}": v.item() for i, v in enumerate(value.reshape(-1))})
        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv, fieldnames=list(row))
            self.csv_writer.writeheader()
        self.csv_writer.writerow(row)
        self.videos["scene"].append_data(np.ascontiguousarray(scene))
        for side in tactile:
            self.videos[f"{side}_tactile"].append_data(images[f"tactile/{side}/rgb"])

        if self.count == 0 or bool(sample["terminated"] | sample["truncated"]):
            name = "first" if self.count == 0 else "terminal"
            imageio.imwrite(self.output / f"{name}_scene.png", scene)
            for side in tactile:
                imageio.imwrite(self.output / f"{name}_{side}.png", images[f"tactile/{side}/rgb"])
        self.count += 1
        self.total_reward += float(sample["reward"])
        self.max_force = max(self.max_force, float(np.linalg.norm(sample["force/contact_local_n"])))
        self.last = sample
        self.pending = None
        if self.count % 30 == 0:
            self.h5.flush()
            self.csv.flush()
            print(f"RECORD_PROGRESS step={step} time={step * env.step_dt:.2f}s "
                  f"success={bool(sample['metrics/episode_succeeded'])}", flush=True)

    def finish(self):
        if self.last is None or not bool(self.last["terminated"] | self.last["truncated"]):
            raise RuntimeError("Recording did not reach an episode boundary")
        if self.pending is not None:
            raise RuntimeError("An action is missing its recorded transition")
        self.h5.attrs["frame_count"] = self.count
        self.h5.attrs["status"] = "complete"
        self.h5.flush()
        return {
            "frames": self.count,
            "recorded_duration_s": float(self.last["time_s"]),
            "completed_trials": 1,
            "successful_trials": int(self.last["metrics/episode_succeeded"]),
            "first_success_time_s": (
                float(self.last["metrics/first_success_step"]) * float(self.last["time_s"]) / self.count
                if bool(self.last["metrics/episode_succeeded"]) else None
            ),
            "total_reward": self.total_reward,
            "max_processed_contact_force_norm_n": self.max_force,
            "max_press_depth_mm": self.max_press,
            "terminal_frame_recorded_before_reset": True,
        }

    def close(self):
        self.stack.close()
