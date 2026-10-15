"""Verify world-vertical pegs and one-axis labels, then export real RGB previews."""
import argparse, json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tactile_pose.normal_sampling import SCHEMA, ZERO_ROTATION_TO
from tactile_pose.normal_learning import resolve_paths


def inspect(path, output):
    with h5py.File(path) as f:
        assert f.attrs["schema"] == SCHEMA and f.attrs["status"] == "complete"
        meta = json.loads(f.attrs["metadata_json"])
        valid = f["quality/sample_valid"][:]
        assert valid.any() and np.array_equal(valid, f["quality/status_code"][:] == 0)
        n = len(valid)
        rt = Rotation.from_quat(f["tcp/quaternion_wxyz"][:][:, [1,2,3,0]]).as_matrix()
        ro = Rotation.from_quat(f["object/quaternion_wxyz"][:][:, [1,2,3,0]]).as_matrix()
        r = rt.transpose(0,2,1) @ ro
        position = np.einsum("nij,nj->ni",rt.transpose(0,2,1),f["object/position_w_m"][:]-f["tcp/position_w_m"][:])
        np.testing.assert_allclose(f["target/rotation_to"][:], r, atol=1e-6)
        np.testing.assert_allclose(f["target/position_to_m"][:], position, atol=1e-6)
        # Independent extraction of the signed planar angle from sine/cosine pairs.
        delta = r @ ZERO_ROTATION_TO.T
        angles = np.rad2deg(np.arctan2(delta[:,0,2]-delta[:,2,0],delta[:,0,0]+delta[:,2,2]))
        labels = f["target/in_hand_angle_deg"][:]
        np.testing.assert_allclose(labels, angles, atol=1e-4)
        np.testing.assert_allclose(labels, f["sampling/requested_angle_deg"][:], atol=.005)
        ideal = Rotation.from_euler("y",angles,degrees=True).as_matrix()
        off_axis = np.rad2deg(Rotation.from_matrix(ideal.transpose(0,2,1) @ delta).magnitude())
        vertical = np.rad2deg(np.arctan2(np.linalg.norm(ro[:,:2,2],axis=1),ro[:,2,2]))
        assert np.max(off_axis) < .005 and np.max(vertical) < .005
        np.testing.assert_allclose(ro, np.tile(np.array(meta["fixed_object_rotation_w"]),(n,1,1)),atol=1e-6)
        np.testing.assert_allclose(rt[:, :, 1],np.tile(rt[0,:,1],(n,1)),atol=1e-5)
        centers = np.einsum("nij,j->ni",r,meta["object_bbox_center_o_m"])+position
        np.testing.assert_allclose(centers,f["sampling/requested_center_t_m"][:],atol=1e-6)
        limits = np.array(meta["robot_joint_limits"])
        joints = f["robot/joint_position"][:]
        assert np.all(joints >= limits[:,0]-1e-6) and np.all(joints <= limits[:,1]+1e-6)
        assert float(f["quality/robot_base_max_change"][:].max()) < 1e-6
        assert len(np.unique(f["random/seed"][:])) == n
        assert len(np.unique(f["ids/contact_pose_id"][:])) == n
        assert len(np.unique(f["time/sim_time_s"][:])) == 1
        assert len(np.unique(f["time/physics_step"][:])) == 1
        assert np.all(np.diff(f["time/unix_ns"][:])>0)
        bins = f["ids/angle_bin"][:]
        counts = np.bincount(bins[valid],minlength=len(meta["angle_edges_deg"])-1)
        assert np.all(counts==meta["samples_per_angle_bin"])
        assert sum(1 for _ in path.with_name("proposals.jsonl").open()) == meta["proposals"]
        max_depth_error = 0.
        for side in ("left","right"):
            peak = f[f"contact/{side}_max_press_mm"][:]
            requested = f["sampling/requested_depth_mm"][:]
            max_depth_error = max(max_depth_error,float(np.max(np.abs(peak[valid]-requested[valid]))))
            assert np.max(np.abs(peak[valid]-requested[valid])) <= meta["depth_tolerance_mm"]
            assert np.min(f[f"contact/{side}_pixels"][:][valid]) >= meta["contact_min_pixels"]
            assert np.max(f[f"contact/{side}_clipped_pixels"][:][valid]) == 0
            for index in range(n):
                press=f[f"tactile/{side}/press_depth_mm"][index]
                depth=f[f"tactile/{side}/camera_depth_mm"][index]
                expected=np.clip(meta["tactile_calibration"]["zero_reference_mm"][side]-depth,0,4.)
                np.testing.assert_allclose(press,expected,atol=1e-4)
                rgb=f[f"tactile/{side}/rgb"][index]
                assert rgb.shape==(240,320,3) and rgb.dtype==np.uint8
        picked=[int(np.flatnonzero(valid & (bins==i))[0]) for i in range(len(counts))]
        fig,axes=plt.subplots(len(picked),3,figsize=(12,3*len(picked)),squeeze=False,layout="constrained")
        for i,row in enumerate(picked):
            scene=path.parent/f"angle_bin_{i}_scene.png"
            frame=np.asarray(Image.open(scene)) if scene.exists() else None
            if frame is not None and np.any(frame): axes[i,0].imshow(frame)
            else: axes[i,0].text(.5,.5,"Scene camera was not ready\nTactile and pose data are available",ha="center",va="center",transform=axes[i,0].transAxes)
            axes[i,0].set_title(f"Upright object | angle {labels[row]:+.2f} deg",fontsize=10)
            axes[i,0].axis("off")
            for j,side in enumerate(("left","right"),start=1):
                axes[i,j].imshow(f[f"tactile/{side}/rgb"][row])
                axes[i,j].set_title(f"{side} | row {row} | {labels[row]:+.2f} deg",fontsize=10)
                axes[i,j].axis("off")
        fig.suptitle(path.parent.name+" | rotation only about grasp normal",fontsize=14)
        fig.savefig(output/f"{path.parent.name}_angles.png",dpi=130);plt.close(fig)
        row=picked[0]
        preview={side:f[f"tactile/{side}/rgb"][row] for side in ("left","right")}
        preview.update(angle=float(labels[row]),row=row)
        return {"status":"passed","task":meta["task"],"path":str(path),"valid_samples":int(valid.sum()),
                "rendered_attempts":n,"angle_range_deg":[float(labels[valid].min()),float(labels[valid].max())],
                "accepted_per_angle_bin":counts.tolist(),"max_world_vertical_error_deg":float(vertical.max()),
                "max_off_axis_rotation_deg":float(off_axis.max()),
                "max_angle_label_error_deg":float(np.max(np.abs(labels-angles))),
                "max_target_depth_error_mm":max_depth_error,"base_change":float(f['quality/robot_base_max_change'][:].max())},preview


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    paths=resolve_paths([args.input])
    if args.output.exists():p.error("Output directory already exists")
    args.output.mkdir(parents=True)
    reports=[];previews=[]
    for path in paths:
        report,preview=inspect(path,args.output);reports.append(report);previews.append(preview)
    fig,axes=plt.subplots(len(paths),2,figsize=(8,3*len(paths)),squeeze=False,layout="constrained")
    for i,(path,preview) in enumerate(zip(paths,previews)):
        for j,side in enumerate(("left","right")):
            axes[i,j].imshow(preview[side]);axes[i,j].axis("off")
            axes[i,j].set_title(f"{path.parent.name} | {side} | angle {preview['angle']:+.2f} deg",fontsize=10)
    fig.suptitle("World-upright pegs | one in-hand angle about grasp normal",fontsize=14)
    fig.savefig(args.output/"stereo_overview.png",dpi=140);plt.close(fig)
    result={"status":"passed","total_valid_samples":sum(r['valid_samples'] for r in reports),"datasets":reports}
    (args.output/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
