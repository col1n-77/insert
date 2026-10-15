"""Predict object-to-gripper rotation from RGB images and measured opening only."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from tactile_pose.data import prepare_image
from tactile_pose.evaluation import load_model
from tactile_calibration import dataset_calibration, require_matching_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="Read only the three input fields from a collected HDF5")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--left_rgb", type=Path)
    parser.add_argument("--right_rgb", type=Path)
    parser.add_argument("--opening_m", type=float, help="Actual sum of the two finger joint positions, meters")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, ckpt = load_model(args.checkpoint, args.device)
    if args.data and any(x is not None for x in (args.left_rgb, args.right_rgb, args.opening_m)):
        parser.error("Choose HDF5 or images/opening, not both")
    if args.data:
        with h5py.File(args.data, "r") as f:
            require_matching_calibration(ckpt.get("tactile_calibration", {}), dataset_calibration(f))
            left = f["tactile/left/rgb"][args.row]
            right = f["tactile/right/rgb"][args.row]
            opening = float(f["gripper/opening_joint_sum_m"][args.row, 0])
    else:
        if args.left_rgb is None or args.right_rgb is None or args.opening_m is None:
            parser.error("Supply --data, or --left_rgb --right_rgb --opening_m")
        left = np.array(Image.open(args.left_rgb).convert("RGB"))
        right = np.array(Image.open(args.right_rgb).convert("RGB"))
        opening = args.opening_m
    if not np.isfinite(opening) or not 0 < opening < 0.081:
        parser.error("Opening must be finite and within the supported Panda joint-sum range (0, 0.081) m")
    with torch.inference_mode():
        matrix = model(prepare_image(left, ckpt["image_size_hw"])[None].to(args.device),
                       prepare_image(right, ckpt["image_size_hw"])[None].to(args.device),
                       torch.tensor([[opening]], device=args.device))[0].cpu().numpy()
    rot = Rotation.from_matrix(matrix)
    quaternion = rot.as_quat()[[3, 0, 1, 2]]
    if quaternion[0] < 0:
        quaternion *= -1
    print(json.dumps({"rotation_go": matrix.tolist(), "quaternion_go_wxyz": quaternion.tolist(),
                      "euler_xyz_deg": rot.as_euler("xyz", degrees=True).tolist(),
                      "convention": "Object coordinates mapped into panda_hand coordinates; Euler angles are display only",
                      "opening_m": opening,
                      "expected_tactile_calibration": ckpt.get("tactile_calibration", {"id": "legacy_shared_28mm"})}, indent=2))


if __name__ == "__main__":
    main()
