"""Evaluate an immutable snapshot of a saved in-hand six-DOF PPO checkpoint."""
import argparse
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
p.add_argument("--training_run", type=Path, help="Original run directory when re-evaluating a saved snapshot")
p.add_argument("--task", default="InsertAnything-LHole-III-Direct-v0")
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seeds", type=int, nargs="+", default=[1001])
p.add_argument("--modes", nargs="+", choices=("stochastic", "deterministic"), default=["stochastic", "deterministic"])
p.add_argument("--output", type=Path)
p.add_argument("--video", action="store_true", help="Record a full scene video before each automatic reset")
p.add_argument("--video_env", type=int, default=0, help="Environment displayed by the scene camera")
p.add_argument("--camera_eye", type=float, nargs=3, default=(0.98, 0.30, 0.34))
p.add_argument("--camera_lookat", type=float, nargs=3, default=(0.60, 0.0, 0.10))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.checkpoint is None or args.output is None or args.num_envs < 1:
    p.error("--checkpoint, --output and positive num_envs are required")
if not 0 <= args.video_env < args.num_envs:
    p.error("--video_env must identify one of --num_envs environments")
if args.video:
    args.enable_cameras = True
source = args.checkpoint.resolve(strict=True)
run_root = args.training_run.resolve(strict=True) if args.training_run else source.parent.parent
saved_env = yaml.load((run_root/"params/env.yaml").read_text(), Loader=yaml.FullLoader)
saved_agent = yaml.safe_load((run_root/"params/agent.yaml").read_text())
if saved_env["initial_in_hand_tilt_deg"] != 10.0 or saved_env["ctrl"]["lock_roll_pitch"]:
    p.error("Expected the upright-gripper, +/-10 degree peg-tilt, unlocked roll/pitch experiment")
if saved_env["seed"] in args.seeds or len(set(args.seeds)) != len(args.seeds):
    p.error("Use distinct evaluation seeds, excluding the training seed")
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
before = source.stat()
snapshot = args.output/"evaluated_checkpoint.pth"
shutil.copy2(source, snapshot)
after = source.stat()
if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
    raise RuntimeError("Training replaced the source checkpoint during the snapshot; retry with a new output directory")
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import InsertAnything.tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from InsertAnything.tasks.direct.insertanything import insertanything_utils as utils


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    torch.set_num_threads(4)
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = int(saved_env["seed"])
    cfg.from_dict(saved_env)
    cfg.seed = args.seeds[0]
    cfg.scene.num_envs = args.num_envs
    if args.device: cfg.sim.device = args.device
    # The original terminal reward callback logs the same success criterion in
    # training mode. Retain it and count episodes ourselves without its CSV writer.
    if cfg.evaluation_mode: raise ValueError("Expected saved training configuration")
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
        params = saved_agent["params"]
        params["seed"] = args.seeds[0]
        params["load_checkpoint"] = True
        params["load_path"] = str(snapshot)
        params["config"].update(num_actors=b.num_envs, device=b.device, device_name=b.device)
        clip_obs = params["env"].get("clip_observations", float("inf"))
        clip_actions = params["env"].get("clip_actions", float("inf"))
        wrapper = RlGamesVecEnvWrapper(env, b.device, clip_obs, clip_actions)
        vecenv.register("IsaacRlgWrapper", lambda config_name,num_actors,**kw: RlGamesGpuEnv(config_name,num_actors,**kw))
        env_configurations.register("rlgpu", {"vecenv_type":"IsaacRlgWrapper", "env_creator":lambda **kw:wrapper})
        runner=Runner(); runner.load(saved_agent); player=runner.create_player(); player.restore(str(snapshot))
        weights=torch.load(snapshot,map_location="cpu",weights_only=False)
        if not all(not v.is_floating_point() or torch.isfinite(v).all() for v in weights["model"].values()):
            raise ValueError("Non-finite policy weights")
        base={"schema":"insertanything_inhand_full_eval_v1", "task":args.task,
              "source_checkpoint":str(source), "evaluated_checkpoint":str(snapshot),
              "checkpoint_sha256":hashlib.file_digest(snapshot.open("rb"),"sha256").hexdigest(),
              "checkpoint_epoch":weights.get("epoch"), "training_frames":weights.get("frame"),
              "config_source":str(run_root/"params/env.yaml"), "num_envs":b.num_envs,
              "seeds":args.seeds, "initial_in_hand_tilt_deg":cfg.initial_in_hand_tilt_deg,
              "lock_roll_pitch":cfg.ctrl.lock_roll_pitch, "nominal_duration_s":cfg.episode_length_s,
              "step_dt_s":b.step_dt, "observation_space":cfg.observation_space,"state_space":cfg.state_space,
              "native_success_definition":"Original task XY <2.5 mm, depth below height*success_threshold, symmetry-aware yaw < task threshold; any successful step in a full episode",
              "strict_success_definition":"Native success AND depth >= -2 mm AND ground-truth peg/hole axis angle <3 degrees; no tactile sensor/contact condition",
              "phase":"post_action_pre_reset; policy observations are saved before their corresponding action"}
        if args.video:
            import imageio.v2 as imageio
            base["video"] = {"env_id":args.video_env, "resolution_wh":list(cfg.viewer.resolution),
                             "fps":1.0/b.step_dt, "camera_eye_env_m":list(cfg.viewer.eye),
                             "camera_lookat_env_m":list(cfg.viewer.lookat),
                             "phase":"after action physics, before automatic reset; one frame per control step"}
        del weights
        dump(args.output/"report.json",{**base,"status":"running","results":{}})
        original_reward=b._get_rewards
        current={}
        z=torch.tensor([0.,0.,1.],device=b.device).expand(b.num_envs,3)

        def capture():
            reward=original_reward()
            hp,hq=utils.get_held_base_pose(b.held_pos,b.held_quat,b.num_envs,b.device)
            tp,tq=utils.get_target_held_base_pose(b.fixed_pos,b.fixed_quat,b.cfg_task,b.num_envs,b.device)
            ao=quat_apply(b.held_quat,z);ah=quat_apply(b.fixed_quat,z)
            angle=torch.rad2deg(torch.acos((ao*ah).sum(-1).clamp(-1,1)))
            native=b._get_curr_successes(b.cfg_task.success_threshold)
            dz=(hp[:,2]-tp[:,2])*1000
            _,_,yaw=utils.get_closest_symmetry_transform(hq,tq,[np.deg2rad(x) for x in b.cfg_task.symmetry_angles_deg])
            at=quat_apply_inverse(b.fingertip_midpoint_quat,ao)
            values={"native_success":native,"strict_success":native&(dz>=-2)&(angle<3),
                    "axis_error_deg":angle,"xy_error_mm":torch.linalg.vector_norm(hp[:,:2]-tp[:,:2],dim=-1)*1000,
                    "z_error_mm":dz,"yaw_error_deg":torch.rad2deg(yaw),
                    "in_hand_angle_deg":torch.rad2deg(torch.atan2(-at[:,0],-at[:,2])),
                    "tcp_tilt_deg":torch.rad2deg(torch.acos((-quat_apply(b.fingertip_midpoint_quat,z)[:,2]).clamp(-1,1))),
                    "contact_force_norm_n":torch.linalg.vector_norm(b.contact_force_local,dim=-1)}
            current["diag"]={k:v.cpu().numpy().copy() for k,v in values.items()}
            state={"tcp_pos_w":b.fingertip_midpoint_pos+b.scene.env_origins,"tcp_quat_wxyz":b.fingertip_midpoint_quat,
                   "object_pos_w":b.held_pos+b.scene.env_origins,"object_quat_wxyz":b.held_quat,
                   "hole_pos_w":b.fixed_pos+b.scene.env_origins,"hole_quat_wxyz":b.fixed_quat,
                   "joint_pos":b.joint_pos,"joint_vel":b.joint_vel,"contact_force_local_n":b.contact_force_local,
                   "filtered_actions":b.actions}
            current["state"]={k:v.cpu().numpy().copy() for k,v in state.items()}
            if video_writer is not None:
                # Refresh graphics at the final physics state without another
                # physics step or noisy observation call. The terminal frame
                # must be captured here, before DirectRLEnv resets the scene.
                b.sim.forward()
                b.sim.render()
                b.sim.render()
                scene = env.render()
                if scene is None or scene.shape != (720, 1280, 3) or not np.any(scene):
                    raise RuntimeError("Scene camera produced an empty or unexpected frame")
                video_writer.append_data(np.ascontiguousarray(scene))
                e = args.video_env
                step = int(b.episode_length_buf[e])
                if step == 1 or step % 100 == 0 or bool(b.reset_terminated[e] | b.reset_time_outs[e]):
                    imageio.imwrite(folder/f"scene_step_{step:04d}.png", scene)
                if bool(current["diag"]["strict_success"][e]) and not current.get("video_success_saved", False):
                    imageio.imwrite(folder/"first_strict_success.png", scene)
                    current["video_success_saved"] = True
            return reward
        b._get_rewards=capture
        results={}
        with torch.inference_mode():
            for mode in args.modes:
                episodes=[]
                for seed in args.seeds:
                    folder=args.output/f"{mode}_seed_{seed}";folder.mkdir()
                    obs,_=env.reset(seed=seed)
                    player.get_batch_size(obs["policy"],1);player.reset()
                    dump(folder/"reset.json",{"sampled_angle_deg":torch.rad2deg(b.initial_in_hand_tilt).tolist(),
                         "tcp_pos_w":(b.fingertip_midpoint_pos+b.scene.env_origins).tolist(),"tcp_quat_wxyz":b.fingertip_midpoint_quat.tolist(),
                         "object_pos_w":(b.held_pos+b.scene.env_origins).tolist(),"object_quat_wxyz":b.held_quat.tolist()})
                    if args.video:
                        env.render()
                        for _ in range(4): b.sim.render()
                        imageio.imwrite(folder/"reset_scene.png", env.render())
                        current["video_success_saved"] = False
                        video_writer = imageio.get_writer(
                            folder/"scene.mp4", fps=1.0/b.step_dt, codec="libx264",
                            quality=8, macro_block_size=1, ffmpeg_params=["-threads", "2"])
                    rows=[];frames=[];start=time.monotonic()
                    with (folder/"diagnostics.csv").open("w") as file:
                        writer=None
                        for step in range(1,b.max_episode_length+1):
                            actor_obs=obs["policy"].clamp(-clip_obs,clip_obs)
                            actions=player.get_action(player.obs_to_torch(actor_obs),is_deterministic=mode=="deterministic").clamp(-clip_actions,clip_actions)
                            previous_obs=actor_obs.cpu().numpy().copy()
                            obs,reward,terminated,truncated,_=env.step(actions)
                            if not torch.isfinite(reward).all() or not torch.isfinite(actions).all() or not all(torch.isfinite(x).all() for x in obs.values()):
                                raise RuntimeError("Non-finite rollout")
                            frames.append({**current["state"],"raw_actions":actions.cpu().numpy().copy(),"policy_obs_before_action":previous_obs})
                            rew=reward.cpu().numpy()
                            for e in range(b.num_envs):
                                row={"env_id":e,"step":step,"time_s":step*b.step_dt,"reward":float(rew[e]),**{k:float(v[e]) for k,v in current["diag"].items()}}
                                if writer is None: writer=csv.DictWriter(file,fieldnames=list(row));writer.writeheader()
                                writer.writerow(row);rows.append(row)
                            dones=terminated|truncated
                            if step%50==0:
                                file.flush();print(f"BASE_EVAL mode={mode} seed={seed} step={step} elapsed={time.monotonic()-start:.1f}s",flush=True)
                            if dones.any():
                                if not dones.all() or step!=b.max_episode_length-1: raise RuntimeError("Unexpected reset timing")
                                break
                        else: raise RuntimeError("Full episode did not terminate")
                    if video_writer is not None:
                        video_writer.close()
                        video_writer = None
                    np.savez_compressed(folder/"states.npz",**{k:np.stack([f[k] for f in frames]) for k in frames[0]})
                    for e in range(b.num_envs):
                        trajectory=rows[e::b.num_envs]
                        episodes.append({"seed":seed,"env_id":e,"steps":len(trajectory),
                           "native_success":any(r["native_success"] for r in trajectory),"strict_success":any(r["strict_success"] for r in trajectory),
                           "terminal_native_success":bool(trajectory[-1]["native_success"]),"terminal_strict_success":bool(trajectory[-1]["strict_success"]),
                           "first_native_success_s":next((r["time_s"] for r in trajectory if r["native_success"]),None),
                           "first_strict_success_s":next((r["time_s"] for r in trajectory if r["strict_success"]),None),
                           "return":sum(r["reward"] for r in trajectory),
                           **{"terminal_"+k:trajectory[-1][k] for k in ("axis_error_deg","xy_error_mm","z_error_mm","yaw_error_deg")}})
                    n=len(episodes)
                    results[mode]={"completed_episodes":n,"native_success_episodes":sum(e["native_success"] for e in episodes),
                                   "strict_success_episodes":sum(e["strict_success"] for e in episodes),
                                   "terminal_native_success_episodes":sum(e["terminal_native_success"] for e in episodes),
                                   "terminal_strict_success_episodes":sum(e["terminal_strict_success"] for e in episodes),
                                   "mean_return":float(np.mean([e["return"] for e in episodes])),"episodes":episodes}
                    dump(args.output/"report.json",{**base,"status":"running","results":results})
                    print(f"BASE_EVAL_BATCH mode={mode} native={results[mode]['native_success_episodes']}/{n} strict={results[mode]['strict_success_episodes']}/{n}",flush=True)
        dump(args.output/"report.json",{**base,"status":"completed","results":results})
        print("BASE_EVAL_OUTPUT "+str(args.output),flush=True)
    finally:
        if video_writer is not None: video_writer.close()
        env.close()


if __name__=="__main__":
    try: main()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    app.close()
