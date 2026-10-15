"""No-tactile PPO with in-hand angle, axis penalty and opt-in depth-axis gate."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--task", default="InsertAnythingAngle-LHole-III-Direct-v0")
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--max_iterations", type=int, default=1000)
p.add_argument("--horizon", type=int, default=128)
p.add_argument("--minibatch_size", type=int, default=512)
p.add_argument("--seq_length", type=int, default=128)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--angle_noise_std_deg", type=float, default=0.)
p.add_argument("--align_weight", type=float, default=1.)
p.add_argument("--align_sigma_deg", type=float, default=5.)
p.add_argument("--depth_axis_gate", action="store_true", help="Gate only the dense depth reward by true object/hole axis angle")
p.add_argument("--depth_gate_tolerance_deg", type=float, default=3.)
p.add_argument("--depth_gate_sigma_deg", type=float, default=5.)
p.add_argument("--depth_gate_floor", type=float, default=.1)
p.add_argument("--initial_tilt_deg", type=float, default=10.)
p.add_argument("--learning_rate", type=float, default=1.e-4)
p.add_argument("--save_frequency", type=int, default=100)
p.add_argument("--run_name")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if min(args.num_envs, args.max_iterations, args.horizon, args.minibatch_size, args.seq_length, args.save_frequency) < 1:
    p.error("Batch sizes, iterations and save_frequency must be positive")
if (args.num_envs*args.horizon % args.minibatch_size or args.horizon % args.seq_length
        or args.minibatch_size % args.seq_length):
    p.error("Batch/minibatch/horizon must be divisible by minibatch/sequence lengths")
if not args.task.startswith("InsertAnythingAngle-"):
    p.error("Use an independent InsertAnythingAngle task")
if args.angle_noise_std_deg < 0 or args.initial_tilt_deg < 0 or args.align_weight < 0:
    p.error("Noise, tilt range and reward weight must be nonnegative")
if args.align_sigma_deg <= 0 or args.learning_rate <= 0:
    p.error("Reward sigma and learning rate must be positive")
if not (0 <= args.depth_gate_tolerance_deg < 180 and 0 < args.depth_gate_sigma_deg < float("inf")
        and 0 <= args.depth_gate_floor <= 1):
    p.error("Depth gate requires tolerance in [0,180), finite positive sigma, and floor in [0,1]")
app = AppLauncher(args).app

import gymnasium as gym
import torch
import insertanything_angle
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from InsertAnything.tasks.direct.insertanything import insertanything_env as base_module


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def snapshot_sources(output):
    project = Path(__file__).resolve().parents[1]
    roots = {"angle": project, "base_task": Path(base_module.__file__).parent}
    manifest = {}
    for label, root in roots.items():
        paths = ([*root.joinpath("source").rglob("*.py"), *root.joinpath("scripts").glob("*.py"), root/"scripts/python.sh"]
                 if label == "angle" else [*root.glob("*.py"), *root.joinpath("agents").glob("*.yaml")])
        for src in sorted(paths):
            rel = Path(label)/src.relative_to(root)
            dest = output/"source_snapshot"/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            manifest[str(rel)] = {"source": str(src), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    dump(output/"source_manifest.json", manifest)


def main():
    torch.set_num_threads(4)
    device = args.device or "cuda:0"
    cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.initial_in_hand_tilt_deg = args.initial_tilt_deg
    cfg.ctrl.lock_roll_pitch = False
    cfg.angle_obs.noise_std_deg = args.angle_noise_std_deg
    cfg.axis_alignment.reward_weight = args.align_weight
    cfg.axis_alignment.sigma_deg = args.align_sigma_deg
    cfg.axis_alignment.zero_at_alignment = True
    cfg.depth_axis_gate.enabled = args.depth_axis_gate
    cfg.depth_axis_gate.tolerance_deg = args.depth_gate_tolerance_deg
    cfg.depth_axis_gate.sigma_deg = args.depth_gate_sigma_deg
    cfg.depth_axis_gate.min_gate = args.depth_gate_floor
    agent = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    agent["params"].update(seed=args.seed, load_checkpoint=False, load_path="")
    c = agent["params"]["config"]
    root = Path("logs/rl_games/InsertAnythingAngle").resolve()
    name = args.run_name or f"LHole_III_w{args.align_weight:g}_noise{args.angle_noise_std_deg:g}_seed{args.seed}_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if Path(name).name != name: raise ValueError("run_name must be a directory name")
    output = root/name
    output.mkdir(parents=True, exist_ok=False)
    c.update(name="InsertAnythingAngle", device=device, device_name=device, num_actors=args.num_envs,
             max_epochs=args.max_iterations, horizon_length=args.horizon,
             minibatch_size=args.minibatch_size, seq_length=args.seq_length,
             learning_rate=args.learning_rate, lr_schedule="constant", save_frequency=args.save_frequency,
             train_dir=str(root), full_experiment_name=name)
    c["central_value_config"].update(minibatch_size=args.minibatch_size,
             learning_rate=args.learning_rate, lr_schedule="constant")
    contract = {"schema": "insertanything_angle_rl_v3", "task": args.task, "seed": args.seed,
                "status": "initializing", "started_at": datetime.now().astimezone().isoformat(),
                "argv": sys.argv, "max_iterations": args.max_iterations,
                "training_from_scratch": True, "tactile_sensors": False,
                "observation_contract": "original actor prefix, then (actual in-hand angle + independent Gaussian noise)/15deg; critic prefix then true angle/15deg",
                "angle_noise_std_deg": args.angle_noise_std_deg,
                "reward_contract": ("original reward + w*(exp(-(true peg/hole axis angle/sigma)^2)-1)"
                                    + (" + (depth_axis_gate-1)*original_weighted_dense_depth_reward" if args.depth_axis_gate else "")),
                "reward_variant": "axis_depth_gated_v1" if args.depth_axis_gate else "axis_additive_v2",
                "depth_axis_gate": cfg.depth_axis_gate.to_dict(),
                "axis_reward_form": "gaussian_minus_one",
                "align_weight": args.align_weight, "align_sigma_deg": args.align_sigma_deg,
                "initial_in_hand_tilt_deg": args.initial_tilt_deg, "initial_tcp": "original upright wrist",
                "lock_roll_pitch": False, "learning_rate": args.learning_rate,
                "lr_schedule": "constant (actor and central critic)",
                "batch_transitions": args.num_envs*args.horizon}
    dump(output/"run.json", contract)
    snapshot_sources(output)
    env = None
    start = time.monotonic()
    try:
        env = gym.make(args.task, cfg=cfg)
        b = env.unwrapped
        if any("gel" in n.lower() or "tactile" in n.lower() for n in b._robot.body_names):
            raise RuntimeError("Unexpected tactile robot body")
        contract.update(status="running", observation_space=cfg.observation_space,
                        state_space=cfg.state_space, action_space=cfg.action_space,
                        robot_body_names=list(b._robot.body_names), step_dt_s=b.step_dt,
                        nominal_episode_length_s=cfg.episode_length_s)
        dump(output/"run.json", contract)
        dump_yaml(str(output/"params/env.yaml"), cfg)
        dump_yaml(str(output/"params/agent.yaml"), agent)
        wrapper = RlGamesVecEnvWrapper(env, device, clip_obs=agent["params"]["env"].get("clip_observations",5.),
                                       clip_actions=agent["params"]["env"].get("clip_actions",1.))
        vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
        env_configurations.register("rlgpu", {"vecenv_type":"IsaacRlgWrapper", "env_creator":lambda **kw:wrapper})
        runner = Runner(IsaacAlgoObserver()); runner.load(agent); runner.reset()
        runner.run({"train":True, "play":False, "sigma":None})
        contract.update(status="completed", finished_at=datetime.now().astimezone().isoformat(),
                        elapsed_s=time.monotonic()-start,
                        checkpoints=[str(x) for x in sorted((output/"nn").glob("*.pth"))])
        dump(output/"run.json", contract)
        print("ANGLE_RL_OUTPUT "+str(output), flush=True)
    except BaseException as exc:
        contract.update(status="failed", error=repr(exc), elapsed_s=time.monotonic()-start)
        dump(output/"run.json", contract)
        raise
    finally:
        if env is not None: env.close()


if __name__ == "__main__":
    try: main()
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    app.close()
