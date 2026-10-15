"""Bounded GPU simulation check with optional JSON results."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="InsertAnything-CircleHole-I-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--report", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps <= 0 or args.num_envs <= 0:
    parser.error("--steps and --num_envs must be positive")
app = AppLauncher(args).app

import gymnasium as gym
import InsertAnything.tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg


def check_finite(value, label):
    if isinstance(value, dict):
        for key, child in value.items():
            check_finite(child, f"{label}.{key}")
    elif isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"Non-finite values in {label}")


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            check_finite(obs, "reset observations")
            completed = 0
            for _ in range(args.steps):
                actions = 0.1 * (2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1)
                obs, reward, terminated, truncated, _ = env.step(actions)
                check_finite(obs, "observations")
                check_finite(reward, "reward")
                check_finite(env.unwrapped.contact_force_local, "contact force")
                completed += int((terminated | truncated).sum().item())
            result = {
                "status": "PASS",
                "task": args.task,
                "num_envs": env.unwrapped.num_envs,
                "steps": args.steps,
                "seed": args.seed,
                "device": str(env.unwrapped.device),
                "observation_shapes": {key: list(value.shape) for key, value in obs.items()},
                "action_shape": list(env.action_space.shape),
                "completed_trials": completed,
                "last_mean_reward": float(reward.mean().item()),
            }
            print("INSERTANYTHING_SMOKE " + json.dumps(result), flush=True)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    app.close()
