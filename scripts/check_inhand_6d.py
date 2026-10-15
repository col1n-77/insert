"""Bounded physical validation of the original robot with an in-hand peg tilt."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="InsertAnything-LHole-III-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=71)
parser.add_argument("--tilt_deg", type=float, default=10.0)
parser.add_argument("--steps_per_axis", type=int, default=5)
parser.add_argument("--legacy", action="store_true", help="Check unchanged zero-tilt, locked-roll/pitch defaults.")
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 2 or args.steps_per_axis < 1:
    parser.error("Use at least two environments and one step per axis")
app = AppLauncher(args).app

import gymnasium as gym
import InsertAnything.tasks  # noqa: F401, E402
import torch
from isaaclab.utils.math import axis_angle_from_quat
from isaaclab_tasks.utils import parse_env_cfg
from isaacsim.core.utils import torch as tq


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    return not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all())


def degrees_list(value):
    return torch.rad2deg(value).cpu().tolist()


def snapshot(base):
    z = torch.tensor([0.0, 0.0, 1.0], device=base.device).repeat(base.num_envs, 1)
    axis_w = tq.quat_apply(base.held_quat, z)
    axis_t = tq.quat_apply(tq.quat_conjugate(base.fingertip_midpoint_quat), axis_w)
    tcp_down_w = tq.quat_apply(base.fingertip_midpoint_quat, -z)
    return {
        "sampled_in_hand_deg": degrees_list(base.initial_in_hand_tilt),
        "actual_in_hand_deg": degrees_list(torch.atan2(-axis_t[:, 0], -axis_t[:, 2])),
        "out_of_plane_deg": degrees_list(torch.asin(axis_t[:, 1].clamp(-1, 1))),
        "peg_world_tilt_deg": degrees_list(torch.acos(axis_w[:, 2].clamp(-1, 1))),
        "tcp_world_tilt_deg": degrees_list(torch.acos(tcp_down_w[:, 2].clamp(-1, 1))),
        "finger_joint_pos_m": base.joint_pos[:, 7:9].cpu().tolist(),
    }


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    if not args.legacy:
        cfg.initial_in_hand_tilt_deg = args.tilt_deg
        cfg.ctrl.lock_roll_pitch = False
    task_before = cfg.task.to_dict()
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    resets = []
    target_rotations = {}
    actual_rotations = {}
    last_target = {}
    original_generate = base.generate_ctrl_signals

    def capture(**kwargs):
        last_target["quat"] = kwargs["ctrl_target_fingertip_midpoint_quat"].clone()
        return original_generate(**kwargs)

    base.generate_ctrl_signals = capture
    try:
        with torch.inference_mode():
            for axis, name in enumerate(("roll", "pitch", "yaw")):
                obs, _ = env.reset()
                assert finite(obs), "Reset observations must be finite"
                resets.append(snapshot(base))

                # Inspect the physical controller target before integrating dynamics.
                base.actions.zero_()
                base.actions[:, axis + 3] = 0.15
                before_q = base.fingertip_midpoint_quat.clone()
                base._apply_action()
                delta = tq.quat_mul(last_target["quat"], tq.quat_conjugate(before_q))
                target_rotations[name] = torch.rad2deg(axis_angle_from_quat(delta)).cpu().tolist()
                if not args.legacy:
                    assert torch.allclose(
                        axis_angle_from_quat(delta), base.actions[:, 3:6] * base.rot_threshold,
                        atol=2e-5, rtol=2e-4,
                    ), f"{name} action was changed by a roll/pitch clamp"
                else:
                    roll, pitch, _ = tq.get_euler_xyz(last_target["quat"])
                    assert bool((torch.abs(roll - torch.pi) < 1e-4).all())
                    assert bool((torch.abs(torch.remainder(pitch + torch.pi, 2 * torch.pi) - torch.pi) < 1e-4).all())

                base.actions.zero_()
                actions = torch.zeros(env.action_space.shape, device=base.device)
                actions[:, axis + 3] = 0.15
                for _ in range(args.steps_per_axis):
                    obs, reward, _, _, _ = env.step(actions)
                    assert finite(obs) and finite(reward), "Non-finite simulated observations or rewards"
                    assert finite(base.joint_pos) and finite(base.contact_force_local)
                delta = tq.quat_mul(base.fingertip_midpoint_quat, tq.quat_conjugate(before_q))
                actual_rotations[name] = torch.rad2deg(axis_angle_from_quat(delta)).cpu().tolist()
                if not args.legacy:
                    assert abs(float(torch.rad2deg(axis_angle_from_quat(delta))[:, axis].mean())) > 0.02, (
                        f"No physical response to the {name} command"
                    )

            # The all-zero rotation must be safe, including during physical grasp closure.
            base.actions.zero_()
            base._apply_action()
            assert finite(last_target["quat"])
            base.close_gripper_in_place()
            assert finite(last_target["quat"])
            assert base.cfg.task.to_dict() == task_before, "Per-task settings were modified"
            assert not any("gelsight" in name or "gelpad" in name for name in base._robot.body_names)

            sampled = torch.tensor([v for reset in resets for v in reset["sampled_in_hand_deg"]])
            actual = torch.tensor([v for reset in resets for v in reset["actual_in_hand_deg"]])
            if not args.legacy and args.tilt_deg > 0:
                assert float(sampled.abs().max()) <= args.tilt_deg + 1e-5
                assert float(actual.std()) > 0.2, "Physical closure erased the peg tilt"
                assert not torch.equal(torch.tensor(resets[0]["sampled_in_hand_deg"]), torch.tensor(resets[1]["sampled_in_hand_deg"]))
            else:
                assert bool((sampled == 0).all())

            result = {
                "status": "passed", "task": args.task, "seed": args.seed,
                "num_envs": base.num_envs, "lock_roll_pitch": cfg.ctrl.lock_roll_pitch,
                "initial_in_hand_tilt_deg": cfg.initial_in_hand_tilt_deg,
                "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
                "action_shape": list(env.action_space.shape),
                "robot_body_names": base._robot.body_names,
                "task_config_unchanged": True, "resets": resets,
                "closure_angle_drift_mae_deg": float((actual - sampled).abs().mean()),
                "target_world_rotation_vectors_deg": target_rotations,
                "actual_world_rotation_vectors_deg": actual_rotations,
                "steps_per_axis": args.steps_per_axis,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n")
            print("INHAND_6D_CHECK " + json.dumps(result), flush=True)
    finally:
        base.generate_ctrl_signals = original_generate
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
