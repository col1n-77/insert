"""Check live calibrated optics on a symmetric peg and on an unloaded gripper."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import insertanything_tactile
from isaaclab_tasks.utils import parse_env_cfg
from tactile_calibration import height_from_depth_m


def cpu(value):
    return value[0].detach().cpu().numpy().copy()


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    task = "InsertAnythingTactile-SquareHole-II-Direct-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=1)
    cfg.seed = 0
    cfg.evaluation_mode = True
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    report = {"status": "running", "task": task, "seed": 0, "steps": 8,
              "tactile_calibration": base.tactile_calibration, "sides": {}}
    arrays = {}
    try:
        with torch.inference_mode():
            env.reset()
            for _ in range(8):
                env.step(torch.zeros_like(base.actions))
            observations = base.refresh_tactile()
            masks = []
            for side in ("left", "right"):
                sensor = base.tactile_sensors[side]
                depth = sensor.camera.data.output["depth"][..., 0]
                zero = base.tactile_calibration["zero_reference_mm"][side]
                expected = height_from_depth_m(depth, zero, 4.0, cfg.tactile.clipping_range)
                torch.testing.assert_close(observations[side]["press_depth_mm"], -expected, atol=1e-6, rtol=0)
                optical = sensor.optical_simulator._taxim.render_direct(
                    expected, with_shadow=False, press_depth=(-expected).amax((1, 2)), orig_hm_fmt=False).movedim(1, 3)
                torch.testing.assert_close(observations[side]["rgb"], sensor._as_uint8_rgb(optical), atol=0, rtol=0)
                legacy = height_from_depth_m(depth, 28.0, 4.0, cfg.tactile.clipping_range)
                legacy_rgb = sensor._as_uint8_rgb(sensor.optical_simulator._taxim.render_direct(
                    legacy, with_shadow=False, press_depth=(-legacy).amax((1, 2)), orig_hm_fmt=False).movedim(1, 3))
                press = cpu(observations[side]["press_depth_mm"])
                masks.append(press > 0.02)
                for name, value in (("calibrated_rgb", observations[side]["rgb"]),
                                    ("legacy_rgb", legacy_rgb), ("press_mm", -expected),
                                    ("camera_depth_mm", depth * 1000)):
                    arrays[f"{side}_{name}"] = cpu(value)
                    if name.endswith("rgb"):
                        Image.fromarray(cpu(value)).save(args.output / f"{side}_{name}.png")
                report["sides"][side] = {"max_press_mm": float(press.max()), "contact_pixels": int(masks[-1].sum()),
                                         "legacy_max_press_mm": float((-legacy).max()),
                                         "rgb_max_error_vs_independent_render": 0}
            mirror = np.fliplr(masks[1])
            report["mirrored_contact_iou"] = float(np.logical_and(masks[0], mirror).sum() / np.logical_or(masks[0], mirror).sum())
            assert report["mirrored_contact_iou"] > 0.9, report
            assert min(s["contact_pixels"] for s in report["sides"].values()) > 1000, report
            # Remove only the held peg, render at the SAME finger poses, and
            # check that each camera's empty view creates no phantom contact.
            state = base._held_asset.data.root_state_w.clone()
            displaced = state.clone()
            displaced[:, 0] += 2.0
            displaced[:, 7:] = 0
            base._held_asset.write_root_state_to_sim(displaced)
            empty = base.refresh_tactile()
            for side in ("left", "right"):
                press = cpu(empty[side]["press_depth_mm"])
                report["sides"][side]["unloaded_max_press_mm"] = float(press.max())
                report["sides"][side]["unloaded_contact_pixels"] = int((press > 0.02).sum())
                assert (press > 0.02).sum() == 0, (side, float(press.max()))
                arrays[f"{side}_unloaded_rgb"] = cpu(empty[side]["rgb"])
                Image.fromarray(arrays[f"{side}_unloaded_rgb"]).save(args.output / f"{side}_unloaded_rgb.png")
            base._held_asset.write_root_state_to_sim(state)
        np.savez_compressed(args.output / "validation.npz", **arrays)
        fig, axes = plt.subplots(3, 2, figsize=(8, 8), layout="constrained")
        for row, (kind, label) in enumerate((("legacy_rgb", "Shared 28 mm (same depth)"),
                                           ("calibrated_rgb", "Per-side calibrated runtime"),
                                           ("unloaded_rgb", "Unloaded gripper"))):
            for col, side in enumerate(("left", "right")):
                axes[row, col].imshow(arrays[f"{side}_{kind}"])
                axes[row, col].set_title(f"{label} | {side}")
                axes[row, col].axis("off")
        fig.savefig(args.output / "calibration_validation.png", dpi=160)
        plt.close(fig)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        env.close()
    print("CALIBRATION_REPORT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
