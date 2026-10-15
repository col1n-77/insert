"""Predict one signed angle from a saved stereo tactile observation."""
import argparse, json
from pathlib import Path
import h5py
import torch
from tactile_calibration import dataset_calibration, require_matching_calibration
from tactile_pose.data import prepare_image
from tactile_pose.normal_sampling import SCHEMA
from tactile_pose.normal_learning import CHECKPOINT_SCHEMA, NormalAngleNet


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--h5",type=Path,required=True)
    p.add_argument("--row",type=int,default=0)
    p.add_argument("--show_label",action="store_true")
    args=p.parse_args();torch.set_num_threads(4)
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if ck.get("schema")!=CHECKPOINT_SCHEMA:raise ValueError("Expected a v4 one-angle checkpoint")
    model=NormalAngleNet(**ck["config"]["model"]);model.load_state_dict(ck["model_state"]);model.eval()
    with h5py.File(args.h5) as f:
        if f.attrs.get("schema")!=SCHEMA:raise ValueError("Expected a v4 single-normal sample")
        require_matching_calibration(ck["tactile_calibration"],dataset_calibration(f))
        metadata=json.loads(f.attrs["metadata_json"])
        if any(metadata[k]!=v for k,v in ck["target_contract"].items()):raise ValueError("Target angle convention mismatch")
        if args.row<0 or args.row>=len(f["quality/sample_valid"]):raise ValueError("Row out of range")
        if not f["quality/sample_valid"][args.row]:raise ValueError("Sample has invalid contact/pose quality")
        images=[prepare_image(f[f"tactile/{side}/rgb"][args.row],ck["config"]["image_size_hw"])[None] for side in ("left","right")]
        opening=torch.tensor(f["gripper/opening_joint_sum_m"][args.row])[None]
        with torch.inference_mode():prediction=float(model(*images,opening)[0])
        result={"predicted_angle_deg":prediction,"axis":"+TCP Y (grasp normal)","row":args.row}
        if args.show_label:
            truth=float(f["target/in_hand_angle_deg"][args.row])
            result.update(actual_angle_deg=truth,absolute_error_deg=abs(prediction-truth))
    print(json.dumps(result))


if __name__=="__main__":main()
