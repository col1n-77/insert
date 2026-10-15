"""Read-only geometry diagnosis; counterfactual optical calibration stays in memory."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
args.enable_cameras=True
app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
from PIL import Image
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation
import torch
import insertanything_tactile
from isaaclab_tasks.utils import parse_env_cfg


def cpu(value):
    return value[0].detach().cpu().numpy().copy()


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()


def rgb_from_height(sensor,height):
    press=(-height).amax((1,2))
    value=sensor.optical_simulator._taxim.render_direct(height,with_shadow=False,
                                                      press_depth=press,orig_hm_fmt=False).movedim(1,3)
    return cpu(sensor._as_uint8_rgb(value))


def main():
    args.output.mkdir(parents=True,exist_ok=False)
    task='InsertAnythingTactile-SquareHole-II-Direct-v0'
    cfg=parse_env_cfg(task,device=args.device or 'cuda:0',num_envs=1)
    cfg.seed=0
    cfg.evaluation_mode=True
    cfg.tactile.sides=('left','right')
    # Preserve this historical diagnostic's shared-zero vs per-side comparison.
    cfg.tactile.calibration_mode='legacy_shared'
    env=gym.make(task,cfg=cfg)
    base=env.unwrapped
    try:
        with torch.inference_mode():
            env.reset()
            for _ in range(8):
                env.step(torch.zeros_like(base.actions))
            observations=base.refresh_tactile()
            cache=UsdGeom.XformCache()
            stage=base.sim.stage
            report={'task':task,'seed':0,'steps':8,'status':'complete',
                    'changes':'Counterfactual optical zero reference only; no source, asset, or physical pose edits',
                    'sides':{}}
            arrays={}
            directions={}
            bodydata=base._robot.data
            for side in ('left','right'):
                sensor=base.tactile_sensors[side]
                gel_path=f'/World/envs/env_0/Robot/gelpad_{side}'
                gel_prim=stage.GetPrimAtPath(gel_path)
                camera_prim=stage.GetPrimAtPath(f'/World/envs/env_0/Robot/gelsight_mini_case_{side}/Camera')
                inverse_camera=cache.GetLocalToWorldTransform(camera_prim).GetInverse()
                camera_vertices=[]
                for prim in Usd.PrimRange(gel_prim):
                    if prim.IsA(UsdGeom.Mesh):
                        mesh_to_camera=cache.GetLocalToWorldTransform(prim)*inverse_camera
                        points=UsdGeom.Mesh(prim).GetPointsAttr().Get()
                        camera_vertices.extend([list(mesh_to_camera.Transform(Gf.Vec3d(*map(float,p)))) for p in points])
                assert camera_vertices,gel_path
                body=base._robot.body_names.index(f'gelpad_{side}')
                gel_position=cpu(bodydata.body_pos_w[:,body])
                gel_rotation=rotation(cpu(bodydata.body_quat_w[:,body]))
                camera_position=cpu(sensor.camera.data.pos_w)
                camera_rotation=rotation(cpu(sensor.camera.data.quat_w_opengl))
                # Both transforms come from the same authored USD frame. The
                # runtime physics may use Fabric while camera pose metadata
                # remains at its authored world pose; never mix those frames.
                vertices_camera=np.asarray(camera_vertices)
                # Camera depth is distance to its image plane along -Z.
                gel_depths=-vertices_camera[:,2]*1000
                zero_mm=float(gel_depths.max())
                assert 20 < zero_mm < 35, (side,zero_mm)
                depth=cpu(observations[side]['camera_depth_mm'])
                original_press=cpu(observations[side]['press_depth_mm'])
                calibrated_height=np.clip(depth-zero_mm,-4,0).astype(np.float32)
                calibrated_press=-calibrated_height
                corrected=rgb_from_height(sensor,torch.from_numpy(calibrated_height[None]).to(base.device))
                original=cpu(observations[side]['rgb'])
                for kind,image in (('original',original),('calibrated',corrected)):
                    Image.fromarray(image).save(args.output/f'{kind}_{side}.png')
                directions[side]={'forward':-camera_rotation[:,2],'up':camera_rotation[:,1]}
                arrays.update({f'{side}_original_rgb':original,f'{side}_calibrated_rgb':corrected,
                               f'{side}_camera_depth_mm':depth,f'{side}_original_press_mm':original_press,
                               f'{side}_calibrated_press_mm':calibrated_press})
                report['sides'][side]={
                    'camera_to_gel_mesh_front_mm':zero_mm,
                    'camera_to_gel_mesh_back_mm':float(gel_depths.min()),
                    'shared_optical_zero_mm':(sensor.cfg.sensor_camera_cfg.clipping_range[0]+sensor.cfg.gelpad_dimensions.height)*1000,
                    'original_max_press_mm':float(original_press.max()),
                    'calibrated_max_press_mm':float(calibrated_press.max()),
                    'original_contact_pixels':int((original_press>0.02).sum()),
                    'calibrated_contact_pixels':int((calibrated_press>0.02).sum()),
                    'camera_authored_position_w_m':camera_position.tolist(),
                    'gel_runtime_position_w_m':gel_position.tolist(),
                    'camera_authored_quaternion_opengl_wxyz':cpu(sensor.camera.data.quat_w_opengl).tolist(),
                }
            synthetic=torch.zeros((1,240,320),device=base.device)
            synthetic[:,80:160,110:210]=-0.15
            controls=[rgb_from_height(base.tactile_sensors[side],synthetic.clone()) for side in ('left','right')]
            report['identical_height_map_rgb_max_difference']=int(np.abs(controls[0].astype(int)-controls[1].astype(int)).max())
            report['camera_forward_dot']=float(directions['left']['forward']@directions['right']['forward'])
            report['camera_up_dot']=float(directions['left']['up']@directions['right']['up'])
            report['actual_finger_joint_position_mm']=(cpu(bodydata.joint_pos)[-2:]*1000).tolist()
            report['geometry_zero_difference_mm']=report['sides']['left']['camera_to_gel_mesh_front_mm']-report['sides']['right']['camera_to_gel_mesh_front_mm']
            np.savez_compressed(args.output/'diagnostic.npz',**arrays)
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print('SYMMETRY_REPORT '+json.dumps(report),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close()
