"""Render one registered tactile scene and synchronous stereo tactile images."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', required=True)
parser.add_argument('--steps', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--tactile_calibration', choices=('geometry_per_side', 'legacy_shared'), default='geometry_per_side')
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1:
    parser.error('--steps must be positive')
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import insertanything_tactile
from isaaclab_tasks.utils import parse_env_cfg


def cpu(tensor):
    return tensor[0].detach().cpu().numpy().copy()


def take_view(env, eye, target, path):
    base = env.unwrapped
    base.sim.set_camera_view(eye=tuple(eye), target=tuple(target),
                             camera_prim_path=base.cfg.viewer.cam_prim_path)
    env.render()  # Create the render product on first use.
    for _ in range(8):
        base.sim.render()  # Render only; the physical state is unchanged.
    frame = env.render()
    if frame is None or not frame.size or not np.any(frame):
        raise RuntimeError(f'Empty scene image: {path}')
    Image.fromarray(frame).save(path)
    return {'eye_w_m': list(map(float, eye)), 'target_w_m': list(map(float, target))}


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(args.task, device=args.device or 'cuda:0', num_envs=1)
    cfg.seed = args.seed
    cfg.evaluation_mode = True
    cfg.tactile.sides = ('left', 'right')
    cfg.tactile.calibration_mode = args.tactile_calibration
    cfg.viewer.resolution = (1280, 960)
    env = gym.make(args.task, cfg=cfg, render_mode='rgb_array')
    base = env.unwrapped
    try:
        with torch.inference_mode():
            env.reset()
            for _ in range(args.steps):
                _, _, terminated, truncated, _ = env.step(torch.zeros_like(base.actions))
                if (terminated | truncated).any():
                    raise RuntimeError('Unexpected automatic reset during scene capture')
            data = base.refresh_tactile()
            arrays = {}
            contact = {}
            for side, values in data.items():
                for name, value in values.items():
                    arrays[f'{side}_{name}'] = cpu(value)
                Image.fromarray(arrays[f'{side}_rgb']).save(args.output / f'{side}_tactile.png')
                press = arrays[f'{side}_press_depth_mm']
                pixels = int(np.count_nonzero(press > 0.02))
                contact[side] = {'max_press_depth_mm': float(press.max()),
                                 'contact_pixels_above_0_02_mm': pixels,
                                 'geometric_contact_valid': pixels >= 64}
            fixed = cpu(base._fixed_asset.data.root_pos_w)
            held = cpu(base._held_asset.data.root_pos_w)
            finger_indices = [base._robot.joint_names.index(name)
                              for name in ('panda_finger_joint1', 'panda_finger_joint2')]
            arrays.update({'fixed_position_w_m': fixed,
                           'fixed_quaternion_wxyz': cpu(base._fixed_asset.data.root_quat_w),
                           'held_position_w_m': held,
                           'held_quaternion_wxyz': cpu(base._held_asset.data.root_quat_w),
                           'robot_joint_position': cpu(base._robot.data.joint_pos),
                           'robot_joint_velocity': cpu(base._robot.data.joint_vel)})
            before = {name: value.copy() for name, value in arrays.items() if '_position' in name}
            fixed_top = fixed + np.array([0, 0, cfg.task.fixed_asset_cfg.height])
            held_mid = held + np.array([0, 0, cfg.task.held_asset_cfg.height * 0.45])
            center = (fixed_top + held_mid) / 2
            center[2] += 0.012
            cameras = {
                'scene': take_view(env, (1.25, 0.8, 1.0), (0.35, 0.0, 0.35),
                                   args.output / 'scene.png'),
                'detail': take_view(env, center + np.array([0.14, 0.10, 0.11]), center,
                                    args.output / 'detail.png'),
                'hole': take_view(env, fixed_top + np.array([0.04, -0.065, 0.10]), fixed_top,
                                  args.output / 'hole.png'),
            }
            assert np.array_equal(before['fixed_position_w_m'], cpu(base._fixed_asset.data.root_pos_w))
            assert np.array_equal(before['held_position_w_m'], cpu(base._held_asset.data.root_pos_w))
            assert np.array_equal(before['robot_joint_position'], cpu(base._robot.data.joint_pos))
            np.savez_compressed(args.output / 'snapshot.npz', **arrays)
            report = {
                'status': 'complete', 'task': args.task, 'seed': args.seed,
                'control_steps_after_reset': args.steps, 'time_after_reset_s': args.steps * base.step_dt,
                'simulation_time_s': float(base.sim.current_time),
                'action_source': 'zero_action_hold',
                'capture_state': 'initialized physical grasp; no insertion policy rollout',
                'same_physical_instant_for_all_views_and_tactile': True,
                'sensor': 'GelSight Mini', 'optics': 'Taxim', 'mechanics': 'PhysX rigid gelpads',
                'contact': contact, 'cameras': cameras,
                'tactile_calibration': base.tactile_calibration,
                'opening_joint_sum_m': float(arrays['robot_joint_position'][finger_indices].sum()),
                'held_asset': {'usd_path': cfg.task.held_asset_cfg.usd_path,
                               'nominal_size_mm': cfg.task.held_asset_cfg.diameter * 1000,
                               'height_mm': cfg.task.held_asset_cfg.height * 1000},
                'fixed_asset': {'usd_path': cfg.task.fixed_asset_cfg.usd_path,
                                'nominal_size_mm': cfg.task.fixed_asset_cfg.diameter * 1000,
                                'height_mm': cfg.task.fixed_asset_cfg.height * 1000},
            }
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print('SCENE_REPORT ' + json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
