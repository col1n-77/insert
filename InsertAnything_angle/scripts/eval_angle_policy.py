"""Evaluate a frozen no-tactile angle-policy checkpoint with physical success metrics.

Each mode/seed runs one complete native episode per parallel environment. State,
reward and diagnostic records are captured after action physics and before the
native automatic reset; observations are stored before their corresponding action.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import yaml
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=Path)
p.add_argument("--training_run", type=Path, help="Run containing params/env.yaml and params/agent.yaml")
p.add_argument("--task", help="Defaults to the training run's task in run.json")
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seeds", type=int, nargs="+", default=[1001, 2002])
p.add_argument("--modes", nargs="+", choices=("stochastic", "deterministic"), default=["deterministic", "stochastic"])
p.add_argument("--noise_std_deg", type=float, help="Explicit evaluation-only angle-noise override; omitted restores saved value")
p.add_argument("--output", type=Path)
p.add_argument("--video", action="store_true")
p.add_argument("--video_env", type=int, default=0)
p.add_argument("--camera_eye", type=float, nargs=3, default=(0.98, 0.30, 0.34))
p.add_argument("--camera_lookat", type=float, nargs=3, default=(0.60, 0.0, 0.10))
# AppLauncher parses the parser during registration, before the final parse.
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.checkpoint is None or args.output is None or args.num_envs < 1:
    p.error("--checkpoint, --output and positive --num_envs are required")
if not 0 <= args.video_env < args.num_envs:
    p.error("--video_env must be in [0, num_envs)")
if args.noise_std_deg is not None and not (0 <= args.noise_std_deg < float("inf")):
    p.error("--noise_std_deg must be finite and nonnegative")
if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
    p.error("Evaluation seeds and modes must not contain duplicates")
if args.video:
    args.enable_cameras = True
source = args.checkpoint.resolve(strict=True)
run_root = args.training_run.resolve(strict=True) if args.training_run else source.parent.parent
saved_env = yaml.load((run_root / "params/env.yaml").read_text(), Loader=yaml.FullLoader)
saved_agent = yaml.safe_load((run_root / "params/agent.yaml").read_text())
saved_contract = json.loads((run_root / "run.json").read_text()) if (run_root / "run.json").is_file() else {}
args.task = args.task or saved_contract.get("task")
if not args.task or not args.task.startswith("InsertAnythingAngle-"):
    p.error("A separate InsertAnythingAngle task is required; provide --task if run.json is missing")
if saved_contract.get("task", args.task) != args.task:
    p.error("--task must match the checkpoint's training task")
if "angle_obs" not in saved_env or "axis_alignment" not in saved_env:
    p.error("Saved environment lacks the no-tactile angle configuration")
# Missing only in historical v1 runs, which used positive Gaussian rewards.
# Resolve the old default explicitly rather than silently changing their reward.
reward_form_inferred = "zero_at_alignment" not in saved_env["axis_alignment"]
saved_env["axis_alignment"].setdefault("zero_at_alignment", False)
depth_gate_inferred_off = "depth_axis_gate" not in saved_env
saved_env.setdefault("depth_axis_gate", {"enabled": False, "tolerance_deg": 3.0,
                                         "sigma_deg": 5.0, "min_gate": 0.1})
if "depth_axis_gate" in saved_contract and saved_contract["depth_axis_gate"] != saved_env["depth_axis_gate"]:
    p.error("Saved run contract and environment disagree on depth-axis gating")
if saved_contract.get("axis_reward_form") == "gaussian_minus_one" and not saved_env["axis_alignment"]["zero_at_alignment"]:
    p.error("Centered-penalty run contract disagrees with saved environment")
if saved_env["ctrl"]["lock_roll_pitch"]:
    p.error("This experiment requires six-DOF control with unlocked roll/pitch")
if saved_env["seed"] in args.seeds:
    p.error("Evaluation seeds must exclude the training seed")
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
snapshot = args.output / "evaluated_checkpoint.pth"
before = source.stat()
shutil.copy2(source, snapshot)
after = source.stat()
if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
    raise RuntimeError("Source checkpoint changed while copying; retry into a fresh output directory")
for name in ("env.yaml", "agent.yaml"):
    shutil.copy2(run_root / "params" / name, args.output / ("training_" + name))
if saved_contract:
    shutil.copy2(run_root / "run.json", args.output / "training_run.json")
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import insertanything_angle  # Registers only the independent angle tasks.
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from InsertAnything.tasks.direct.insertanything import insertanything_utils as utils


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize(episodes):
    result = {"completed_episodes": len(episodes)}
    for key in ("native_success", "strict_success", "terminal_native_success", "terminal_strict_success"):
        count = sum(e[key] for e in episodes)
        result[key + "_episodes"] = count
        result[key + "_rate"] = count / len(episodes)
    for key in ("return", "original_return", "alignment_return", "depth_gate_adjustment_return", "initial_axis_error_deg",
                "terminal_axis_error_deg", "minimum_axis_error_deg", "terminal_xy_error_mm",
                "terminal_z_error_mm", "mean_absolute_angle_noise_deg", "action_saturation_fraction"):
        values = np.asarray([e[key] for e in episodes])
        result["mean_" + key] = float(values.mean())
        if key == "terminal_axis_error_deg":
            result["median_" + key] = float(np.median(values))
            result["p95_" + key] = float(np.percentile(values, 95))
    result["episodes"] = episodes
    return result


def main():
    torch.set_num_threads(4)
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    # configclass.from_dict validates against current field types; seed starts None.
    cfg.seed = int(saved_env["seed"])
    cfg.from_dict(saved_env)
    cfg.seed = args.seeds[0]
    cfg.scene.num_envs = args.num_envs
    if args.device:
        cfg.sim.device = args.device
    if args.noise_std_deg is not None:
        cfg.angle_obs.noise_std_deg = args.noise_std_deg
    if cfg.evaluation_mode:
        raise ValueError("Expected a training configuration; native evaluation CSV side effects must be disabled")
    if args.video:
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = args.video_env
        cfg.viewer.eye = tuple(args.camera_eye)
        cfg.viewer.lookat = tuple(args.camera_lookat)
        cfg.viewer.resolution = (1280, 720)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    b = env.unwrapped
    video_writer = None
    try:
        dump_yaml(str(args.output / "effective_env.yaml"), cfg)
        agent = copy.deepcopy(saved_agent)
        params = agent["params"]
        params["seed"] = args.seeds[0]
        params["load_checkpoint"] = True
        params["load_path"] = str(snapshot)
        params["config"].update(num_actors=b.num_envs, device=b.device, device_name=b.device)
        clip_obs = params["env"].get("clip_observations", float("inf"))
        clip_actions = params["env"].get("clip_actions", 1.)
        wrapper = RlGamesVecEnvWrapper(env, b.device, clip_obs, clip_actions)
        vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapper})
        runner = Runner()
        runner.load(agent)
        player = runner.create_player()
        player.restore(str(snapshot))
        if not torch.all(player.actions_low == -1.) or not torch.all(player.actions_high == 1.):
            raise ValueError("Expected normalized [-1,1] actions for this experiment")
        # Obtain raw Gaussian samples/means, then perform the exact same clipping
        # explicitly. This prevents recording already clipped values as raw actions.
        player.clip_actions = False
        weights = torch.load(snapshot, map_location="cpu", weights_only=False)
        if not all(not v.is_floating_point() or torch.isfinite(v).all() for v in weights["model"].values()):
            raise ValueError("Non-finite checkpoint weights")
        with snapshot.open("rb") as file:
            checkpoint_sha = hashlib.file_digest(file, "sha256").hexdigest()
        base = {
            "schema": "insertanything_angle_eval_v2", "task": args.task,
            "source_checkpoint": str(source), "evaluated_checkpoint": str(snapshot),
            "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": weights.get("epoch"),
            "training_frames": weights.get("frame"), "training_run": str(run_root),
            "training_seed": saved_env["seed"], "seeds": args.seeds, "modes": args.modes,
            "num_envs": b.num_envs, "initial_in_hand_tilt_deg": cfg.initial_in_hand_tilt_deg,
            "lock_roll_pitch": cfg.ctrl.lock_roll_pitch, "nominal_duration_s": cfg.episode_length_s,
            "step_dt_s": b.step_dt, "observation_space": cfg.observation_space,
            "state_space": cfg.state_space, "action_space": cfg.action_space,
            "angle_obs": cfg.angle_obs.to_dict(), "axis_alignment": cfg.axis_alignment.to_dict(),
            "depth_axis_gate": cfg.depth_axis_gate.to_dict(),
            "legacy_depth_gate_inferred_off": depth_gate_inferred_off,
            "noise_std_deg_override": args.noise_std_deg,
            "native_success_definition": "Inherited task criterion, unchanged; success at any step of a complete native episode",
            "strict_success_definition": "Native success AND true peg/hole axis angle < strict_axis_deg AND depth_error_mm >= strict_min_depth_mm",
            "reward_definition": (("Original reward plus reward_weight * (exp(-(true world axis error / sigma_deg)^2)-1)"
                                  if cfg.axis_alignment.zero_at_alignment else
                                  "Original reward plus reward_weight * exp(-(true world axis error / sigma_deg)^2)")
                                  + (" + (depth_axis_gate-1)*original_weighted_dense_depth_reward" if cfg.depth_axis_gate.enabled else "")),
            "reward_variant": "axis_depth_gated_v1" if cfg.depth_axis_gate.enabled else "axis_additive_v2",
            "axis_reward_form": "gaussian_minus_one" if cfg.axis_alignment.zero_at_alignment else "legacy_positive_gaussian",
            "legacy_reward_form_inferred_from_missing_config": reward_form_inferred,
            "phase": "Post-action pre-reset state/diagnostics; policy_obs_before_action is the unclipped preceding observation",
            "angle_noise_phase": "CSV measured angle is post-action; the actor used the previous state angle saved in policy_obs_before_action",
            "action_definition": "raw_actions are Gaussian means (deterministic) or samples (stochastic), clipped_actions are passed to env.step; filtered_actions are EMA control actions",
            "original_reward_comparison": "original_return excludes both axis penalty and depth-gate adjustment; compare physical success across variants, not total return",
        }
        del weights
        if args.video:
            import imageio.v2 as imageio
            base["video"] = {"env_id": args.video_env, "resolution_wh": list(cfg.viewer.resolution),
                             "fps": 1.0 / b.step_dt, "camera_eye_env_m": list(cfg.viewer.eye),
                             "camera_lookat_env_m": list(cfg.viewer.lookat),
                             "phase": "After physics before automatic reset, one frame per control step"}
        dump(args.output / "report.json", {**base, "status": "running", "results": {}})
        original_reward = b._get_rewards
        current = {}
        z = torch.tensor([0., 0., 1.], device=b.device).expand(b.num_envs, 3)
        diagnostic_names = (
            "true_in_hand_angle_deg", "measured_in_hand_angle_deg", "measurement_noise_deg",
            "axis_error_deg", "off_axis_residual_deg", "alignment_reward", "original_reward",
            "depth_axis_gate", "depth_reward_before_axis_gate", "depth_reward_after_axis_gate",
            "depth_gate_reward_adjustment",
            "native_success", "strict_success", "noise_sample_count",
        )

        def capture():
            reward = original_reward()
            signals = b.angle_diagnostics
            hp, hq = utils.get_held_base_pose(b.held_pos, b.held_quat, b.num_envs, b.device)
            tp, tq = utils.get_target_held_base_pose(b.fixed_pos, b.fixed_quat, b.cfg_task, b.num_envs, b.device)
            _, _, yaw = utils.get_closest_symmetry_transform(hq, tq, [np.deg2rad(x) for x in b.cfg_task.symmetry_angles_deg])
            values = {key: signals[key] for key in diagnostic_names}
            values.update(
                xy_error_mm=torch.linalg.vector_norm(hp[:, :2] - tp[:, :2], dim=-1) * 1000,
                z_error_mm=(hp[:, 2] - tp[:, 2]) * 1000,
                yaw_error_deg=torch.rad2deg(yaw),
                tcp_tilt_deg=torch.rad2deg(torch.acos((-quat_apply(b.fingertip_midpoint_quat, z)[:, 2]).clamp(-1, 1))),
                contact_force_norm_n=torch.linalg.vector_norm(b.contact_force_local, dim=-1),
            )
            current["diag"] = {key: value.detach().cpu().numpy().copy() for key, value in values.items()}
            state = {
                "tcp_pos_w": b.fingertip_midpoint_pos + b.scene.env_origins,
                "tcp_quat_wxyz": b.fingertip_midpoint_quat,
                "object_pos_w": b.held_pos + b.scene.env_origins, "object_quat_wxyz": b.held_quat,
                "hole_pos_w": b.fixed_pos + b.scene.env_origins, "hole_quat_wxyz": b.fixed_quat,
                "joint_pos": b.joint_pos, "joint_vel": b.joint_vel,
                "contact_force_local_n": b.contact_force_local, "filtered_actions": b.actions,
                "object_axis_w": signals["object_axis_w"], "hole_axis_w": signals["hole_axis_w"],
                "object_axis_tcp": signals["object_axis_tcp"],
            }
            current["state"] = {key: value.detach().cpu().numpy().copy() for key, value in state.items()}
            if video_writer is not None:
                b.sim.forward()
                b.sim.render()
                b.sim.render()
                scene = env.render()
                if scene is None or scene.shape != (720, 1280, 3) or not np.any(scene):
                    raise RuntimeError("Invalid scene camera frame")
                video_writer.append_data(np.ascontiguousarray(scene))
                e = args.video_env
                step = int(b.episode_length_buf[e])
                if step == 1 or step % 100 == 0 or bool(b.reset_terminated[e] | b.reset_time_outs[e]):
                    imageio.imwrite(folder / f"scene_step_{step:04d}.png", scene)
                if bool(current["diag"]["strict_success"][e]) and not current.get("video_success_saved", False):
                    imageio.imwrite(folder / "first_strict_success.png", scene)
                    current["video_success_saved"] = True
            return reward

        b._get_rewards = capture
        results = {}
        with torch.inference_mode():
            for mode in args.modes:
                episodes = []
                for seed in args.seeds:
                    folder = args.output / f"{mode}_seed_{seed}"
                    folder.mkdir()
                    obs, _ = env.reset(seed=seed)
                    player.get_batch_size(obs["policy"], 1)
                    player.reset()
                    initial = {key: value.detach().cpu().numpy().copy()
                               for key, value in b.get_angle_diagnostics().items()}
                    dump(folder / "reset.json", {
                        "sampled_angle_deg": torch.rad2deg(b.initial_in_hand_tilt).tolist(),
                        "true_in_hand_angle_deg": initial["true_in_hand_angle_deg"].tolist(),
                        "measured_in_hand_angle_deg": initial["measured_in_hand_angle_deg"].tolist(),
                        "axis_error_deg": initial["axis_error_deg"].tolist(),
                        "tcp_pos_w": (b.fingertip_midpoint_pos + b.scene.env_origins).tolist(),
                        "tcp_quat_wxyz": b.fingertip_midpoint_quat.tolist(),
                        "object_pos_w": (b.held_pos + b.scene.env_origins).tolist(),
                        "object_quat_wxyz": b.held_quat.tolist(),
                    })
                    if args.video:
                        env.render()
                        for _ in range(4):
                            b.sim.render()
                        imageio.imwrite(folder / "reset_scene.png", env.render())
                        current["video_success_saved"] = False
                        video_writer = imageio.get_writer(folder / "scene.mp4", fps=1.0 / b.step_dt,
                                                         codec="libx264", quality=8, macro_block_size=1,
                                                         ffmpeg_params=["-threads", "2"])
                    rows, frames = [], []
                    started = time.monotonic()
                    with (folder / "diagnostics.csv").open("w") as file:
                        writer = None
                        for step in range(1, b.max_episode_length + 1):
                            previous_obs = obs["policy"].detach().cpu().numpy().copy()
                            actor_obs = obs["policy"].clamp(-clip_obs, clip_obs)
                            raw_actions = player.get_action(player.obs_to_torch(actor_obs), is_deterministic=mode == "deterministic")
                            # player normally clamps to [-1,1] before wrapper clipping.
                            actions = raw_actions.clamp(-1., 1.).clamp(-clip_actions, clip_actions)
                            obs, reward, terminated, truncated, _ = env.step(actions)
                            if (not torch.isfinite(reward).all() or not torch.isfinite(raw_actions).all()
                                    or not all(torch.isfinite(value).all() for value in obs.values())):
                                raise RuntimeError("Non-finite rollout")
                            raw_np = raw_actions.detach().cpu().numpy().copy()
                            clipped_np = actions.detach().cpu().numpy().copy()
                            frames.append({
                                **current["state"], "raw_actions": raw_np, "clipped_actions": clipped_np,
                                "policy_obs_before_action": previous_obs,
                                "clipped_policy_obs_before_action": actor_obs.detach().cpu().numpy().copy(),
                                "rewards": reward.detach().cpu().numpy().copy(),
                                **{"diag_" + key: value for key, value in current["diag"].items()},
                            })
                            rewards = reward.detach().cpu().numpy()
                            for e in range(b.num_envs):
                                row = {"env_id": e, "step": step, "time_s": step * b.step_dt,
                                       "reward": float(rewards[e]),
                                       "action_saturation_fraction": float(np.mean(np.abs(raw_np[e]) >= 1.)),
                                       **{key: float(value[e]) for key, value in current["diag"].items()}}
                                if writer is None:
                                    writer = csv.DictWriter(file, fieldnames=list(row))
                                    writer.writeheader()
                                writer.writerow(row)
                                rows.append(row)
                            dones = terminated | truncated
                            if step % 50 == 0:
                                file.flush()
                                print(f"ANGLE_EVAL mode={mode} seed={seed} step={step} elapsed={time.monotonic()-started:.1f}s", flush=True)
                            if dones.any():
                                if not dones.all() or step != b.max_episode_length - 1:
                                    raise RuntimeError("Unexpected reset timing; cannot mix independent episodes")
                                break
                        else:
                            raise RuntimeError("Native episode failed to terminate")
                    if video_writer is not None:
                        video_writer.close()
                        video_writer = None
                    np.savez_compressed(folder / "states.npz", **{key: np.stack([frame[key] for frame in frames]) for key in frames[0]})
                    batch_episodes = []
                    for e in range(b.num_envs):
                        trajectory = rows[e::b.num_envs]
                        episode = {
                            "seed": seed, "env_id": e, "steps": len(trajectory),
                            "native_success": any(row["native_success"] for row in trajectory),
                            "strict_success": any(row["strict_success"] for row in trajectory),
                            "terminal_native_success": bool(trajectory[-1]["native_success"]),
                            "terminal_strict_success": bool(trajectory[-1]["strict_success"]),
                            "first_native_success_s": next((row["time_s"] for row in trajectory if row["native_success"]), None),
                            "first_strict_success_s": next((row["time_s"] for row in trajectory if row["strict_success"]), None),
                            "return": sum(row["reward"] for row in trajectory),
                            "original_return": sum(row["original_reward"] for row in trajectory),
                            "alignment_return": sum(row["alignment_reward"] for row in trajectory),
                            "depth_gate_adjustment_return": sum(row["depth_gate_reward_adjustment"] for row in trajectory),
                            "initial_axis_error_deg": float(initial["axis_error_deg"][e]),
                            "initial_true_in_hand_angle_deg": float(initial["true_in_hand_angle_deg"][e]),
                            "minimum_axis_error_deg": min(row["axis_error_deg"] for row in trajectory),
                            "mean_absolute_angle_noise_deg": float(np.mean([abs(row["measurement_noise_deg"]) for row in trajectory])),
                            "action_saturation_fraction": float(np.mean([row["action_saturation_fraction"] for row in trajectory])),
                            **{"terminal_" + key: trajectory[-1][key] for key in (
                                "axis_error_deg", "xy_error_mm", "z_error_mm", "yaw_error_deg",
                                "true_in_hand_angle_deg", "tcp_tilt_deg", "off_axis_residual_deg")},
                        }
                        episodes.append(episode)
                        batch_episodes.append(episode)
                    dump(folder / "summary.json", summarize(batch_episodes))
                    results[mode] = summarize(episodes)
                    dump(args.output / "report.json", {**base, "status": "running", "results": results})
                    print(f"ANGLE_EVAL_BATCH mode={mode} native={results[mode]['native_success_episodes']}/{len(episodes)} strict={results[mode]['strict_success_episodes']}/{len(episodes)}", flush=True)
        dump(args.output / "report.json", {**base, "status": "completed", "results": results})
        print("ANGLE_EVAL_OUTPUT " + str(args.output), flush=True)
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        traceback.print_exc()
        report_path = args.output / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        dump(report_path, report)
        (args.output / "traceback.txt").write_text(traceback.format_exc())
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
