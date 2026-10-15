"""Train InsertAnything PPO with frozen tactile angle feedback and six-DOF control."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--task", default="InsertAnythingTactile6D-LHole-III-Direct-v0")
p.add_argument("--angle_checkpoint", type=Path, default=Path("outputs/dynamic_angle_rl_model_20261012/best.pt"))
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--max_iterations", type=int, default=1000)
p.add_argument("--horizon", type=int, default=64)
p.add_argument("--minibatch_size", type=int, default=None)
p.add_argument("--seq_length", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--align_weight", type=float, default=1.)
p.add_argument("--align_sigma_deg", type=float, default=5.)
p.add_argument("--initial_tilt_deg", type=float, default=10., help="Peg tilt about TCP +Y; wrist starts upright")
p.add_argument("--episode_steps", type=int, default=None, help="Optional shorter episodes for a smoke test")
p.add_argument("--run_name", default=None)
p.add_argument("--checkpoint", type=Path, help="Resume only a policy created by this script with the same angle checkpoint")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if min(args.num_envs, args.max_iterations, args.horizon, args.seq_length) < 1:
    p.error("Batch sizes, horizon and iterations must be positive")
batch = args.num_envs*args.horizon
args.minibatch_size = args.minibatch_size or batch
if (args.minibatch_size < 1 or batch % args.minibatch_size or args.horizon % args.seq_length
        or args.minibatch_size % args.seq_length):
    p.error("Rollout/minibatch/horizon must be divisible by minibatch/sequence lengths")
if not args.angle_checkpoint.is_file(): p.error("Missing --angle_checkpoint")
if args.episode_steps is not None and args.episode_steps < 1: p.error("episode_steps must be positive")
if not args.task.startswith("InsertAnythingTactile6D-"): p.error("Use a separate InsertAnythingTactile6D task")
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
import insertanything_tactile.rl
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def main():
    torch.set_num_threads(4)
    device = args.device or "cuda:0"
    cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.tactile_alignment.checkpoint = str(args.angle_checkpoint.resolve())
    cfg.tactile_alignment.weight = args.align_weight
    cfg.tactile_alignment.sigma_deg = args.align_sigma_deg
    cfg.tactile_alignment.initial_tilt_deg = args.initial_tilt_deg
    if args.episode_steps: cfg.episode_length_s = args.episode_steps*cfg.decimation*cfg.sim.dt
    agent = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    agent["params"]["seed"] = args.seed
    c = agent["params"]["config"]
    root = Path("logs/rl_games/InsertAnythingTactile6D").resolve()
    name = args.run_name or args.task.removeprefix("InsertAnythingTactile6D-")+datetime.now().strftime("_%Y%m%d_%H%M%S_%f")
    if Path(name).name != name: raise ValueError("run_name must be a directory name, not a path")
    output = root/name
    output.mkdir(parents=True, exist_ok=False)
    c.update(device=device, device_name=device, num_actors=args.num_envs,
             max_epochs=args.max_iterations, horizon_length=args.horizon,
             minibatch_size=args.minibatch_size, seq_length=args.seq_length,
             train_dir=str(root), full_experiment_name=name)
    c["central_value_config"]["minibatch_size"] = args.minibatch_size
    env = None
    try:
        env = gym.make(args.task, cfg=cfg)
        base = env.unwrapped
        contract = {"schema": "insertanything_tactile6d_rl_v2", "task": args.task,
                    "initialization": base.initialization_metadata,
                    "observation_space": cfg.observation_space, "state_space": cfg.state_space,
                    "action_space": 6, "action_frame": "world rotation-vector increment",
                    "angle_estimator": base.angle_estimator.metadata, "seed": args.seed,
                    "status": "running", "max_iterations": args.max_iterations}
        if args.checkpoint:
            old = args.checkpoint.resolve(strict=True).parent.parent/"run.json"
            previous = json.loads(old.read_text())
            for key in ("schema", "task", "observation_space", "state_space", "action_space", "initialization"):
                if previous.get(key) != contract[key]: raise ValueError(f"RL resume mismatch: {key}")
            if previous["angle_estimator"]["sha256"] != contract["angle_estimator"]["sha256"]:
                raise ValueError("Cannot silently replace the angle estimator when resuming RL")
            agent["params"].update(load_checkpoint=True, load_path=str(args.checkpoint.resolve()))
        (output/"run.json").write_text(json.dumps(contract, indent=2)+"\n")
        dump_yaml(str(output/"params/env.yaml"), cfg)
        dump_yaml(str(output/"params/agent.yaml"), agent)
        env = RlGamesVecEnvWrapper(env, device, clip_obs=5., clip_actions=1.)
        vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
        runner = Runner(); runner.load(agent); runner.reset()
        options = {"train": True, "play": False, "sigma": None}
        if args.checkpoint: options["checkpoint"] = str(args.checkpoint.resolve())
        runner.run(options)
        contract.update(status="completed", estimator_calls=base.angle_estimator.calls,
                        checkpoints=[str(path) for path in sorted((output/"nn").glob("*.pth"))])
        (output/"run.json").write_text(json.dumps(contract, indent=2)+"\n")
        print("TACTILE_RL_OUTPUT "+str(output), flush=True)
    finally:
        if env is not None: env.close()


if __name__ == "__main__":
    try: main()
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    app.close()
