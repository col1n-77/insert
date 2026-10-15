"""Collect physical grasp episodes for angle-model transfer to the RL environment."""
import argparse, json, os, sys, traceback, time, uuid
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--task",default="InsertAnythingTactile6D-LHole-III-Direct-v0")
p.add_argument("--num_envs",type=int,default=4)
p.add_argument("--grasp_batches",type=int,default=20)
p.add_argument("--seed",type=int,default=110000)
p.add_argument("--initial_tilt_deg",type=float,default=10.,help="Peg tilt about TCP +Y; wrist starts upright")
p.add_argument("--output",type=Path,required=True)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();args.enable_cameras=True
if min(args.num_envs,args.grasp_batches)<1:p.error("Counts must be positive")
app=AppLauncher(args).app
import gymnasium as gym
import h5py
import numpy as np
import torch
import insertanything_tactile.rl
from isaaclab_tasks.utils import parse_env_cfg
from tactile_pose.rl_math import measured_in_hand,quat_apply,quat_inverse,quat_mul


def main():
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=parse_env_cfg(args.task,device=args.device or "cuda:0",num_envs=args.num_envs)
    cfg.seed=args.seed;cfg.episode_length_s=100.
    cfg.tactile_alignment.initial_tilt_deg=args.initial_tilt_deg
    env=gym.make(args.task,cfg=cfg); b=env.unwrapped
    suffix=args.task.removeprefix("InsertAnythingTactile6D-").removesuffix("-Direct-v0")
    object_id=list(insertanything_tactile.rl.TASKS).index(suffix)
    identity=str(uuid.uuid4());count=valid_count=0
    fields={}
    meta={"schema":"tactile_dynamic_normal_v1","dataset_uuid":identity,"task":args.task,
          "initialization":b.initialization_metadata,
          "seed":args.seed,"object_id":object_id,"num_envs":args.num_envs,
          "grasp_batches":args.grasp_batches,"frames_at_control_steps":[0,4,8,12],
          "grouping":"independent physical grasp = dataset_uuid + grasp_id; keep frames together in splits",
          "label":"actual object-in-TCP rotation projected on TCP +Y, degrees; off-axis residual retained",
          "source":"physical reset/closing plus small world-frame six-DOF motions, PhysX rigid gelpads + Taxim",
          "tactile_calibration":b.tactile_calibration,"target_contract":b.angle_estimator.metadata["target_contract"],
          "status":"collecting"}
    try:
        with h5py.File(args.output/"samples.h5","x") as f,torch.inference_mode():
            f.attrs["schema"]=meta["schema"];f.attrs["dataset_uuid"]=identity;f.attrs["status"]="collecting"
            rng=torch.Generator(device=b.device);rng.manual_seed(args.seed+15000)
            for batch in range(args.grasp_batches):
                env.reset()
                actions=torch.empty((b.num_envs,6),device=b.device).uniform_(-.6,.6,generator=rng)
                actions[:,:3]*=.25
                for frame in range(13):
                    if frame>0:env.step(actions if frame<=8 else -actions)
                    if frame not in (0,4,8,12):continue
                    b._ensure_prediction();tactile=b.get_tactile_observations()
                    theta,offaxis=measured_in_hand(b.fingertip_midpoint_quat,b.held_quat)
                    theta=torch.rad2deg(theta);offaxis=torch.rad2deg(offaxis)
                    valid=b.tactile_valid & (theta.abs()<=15) & (offaxis<=3)
                    q_wt=b.fingertip_midpoint_quat
                    row={"ids/env_id":torch.arange(b.num_envs,device=b.device),
                         "ids/grasp_id":torch.arange(b.num_envs,device=b.device)+batch*b.num_envs,
                         "ids/object_id":torch.full((b.num_envs,),object_id,device=b.device,dtype=torch.int32),
                         "time/control_step_in_grasp":torch.full((b.num_envs,),frame,device=b.device,dtype=torch.int32),
                         "time/sim_time_s":np.full(b.num_envs,b.sim.current_time),
                         "time/wall_time_ns":np.full(b.num_envs,time.time_ns(),dtype=np.int64),
                         "random/seed":np.full(b.num_envs,args.seed,dtype=np.int64),
                         "quality/sample_valid":valid,"quality/off_axis_rotation_deg":offaxis,
                         "target/in_hand_angle_deg":theta,
                         "target/quaternion_to_wxyz":quat_mul(quat_inverse(q_wt),b.held_quat),
                         "target/position_to_m":quat_apply(quat_inverse(q_wt),b.held_pos-b.fingertip_midpoint_pos),
                         "tcp/position_w":b.fingertip_midpoint_pos+b.scene.env_origins,
                         "tcp/quaternion_wxyz":q_wt,"object/position_w":b.held_pos+b.scene.env_origins,
                         "object/quaternion_wxyz":b.held_quat,"robot/joint_pos":b._robot.data.joint_pos,
                         "gripper/opening_joint_sum_m":b._robot.data.joint_pos[:,b._finger_joint_ids].sum(-1)}
                    for side,data in tactile.items():
                        row[f"tactile/{side}/rgb"]=data["rgb"]
                        row[f"tactile/{side}/max_depth_mm"]=data["press_depth_mm"].amax((1,2))
                        row[f"tactile/{side}/contact_pixels"]=(data["press_depth_mm"]>.02).sum((1,2))
                    for key,v in row.items():
                        a=v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v
                        if key not in fields:
                            fields[key]=f.create_dataset(key,shape=(0,*a.shape[1:]),maxshape=(None,*a.shape[1:]),
                                chunks=(min(4,b.num_envs),*a.shape[1:]),dtype=a.dtype,compression="gzip",compression_opts=1)
                        d=fields[key];d.resize(count+b.num_envs,axis=0);d[count:count+b.num_envs]=a
                    count+=b.num_envs;valid_count+=int(valid.sum())
                f.flush()
                print(f"DYNAMIC_COLLECTION {suffix} batch={batch+1}/{args.grasp_batches} samples={count} valid={valid_count}",flush=True)
            meta.update(status="complete",samples=count,valid_samples=valid_count)
            f.attrs["metadata_json"]=json.dumps(meta);f.attrs["status"]="complete"
        (args.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    finally:env.close()


if __name__=="__main__":
    try:main()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    app.close()
