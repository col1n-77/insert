"""Fine-tune and evaluate the scalar model on physical grasps, splitting by grasp."""
import argparse,json,random
from pathlib import Path
import h5py,numpy as np,torch
from torch.utils.data import DataLoader,Dataset
from tactile_calibration import require_matching_calibration
from tactile_pose.data import prepare_image
from tactile_pose.normal_learning import NormalAngleNet,CHECKPOINT_SCHEMA,evaluate_angles


class DynamicDataset(Dataset):
    def __init__(self,manifest,split,image_size):
        self.cache=[]
        handles=[h5py.File(p) for p in manifest["files"]]
        try:
            for r in manifest["splits"][split]:
                f,row=handles[r["file"]],r["row"]
                self.cache.append({"left":prepare_image(f["tactile/left/rgb"][row],image_size),
                    "right":prepare_image(f["tactile/right/rgb"][row],image_size),
                    "opening":torch.tensor(np.asarray(f["gripper/opening_joint_sum_m"][row]).reshape(1),dtype=torch.float32),
                    "angle":torch.tensor(float(f["target/in_hand_angle_deg"][row])),"object_id":r["object_id"]})
        finally:
            for f in handles:f.close()
    def __len__(self):return len(self.cache)
    def __getitem__(self,i):return self.cache[i]


def manifest_for(paths,ck,seed,external=False):
    files=[]
    for path in paths:
        if path.is_file():files.append(path.resolve())
        else:files.extend(sorted(path.resolve().rglob("samples.h5")))
    if not files or len(set(files))!=len(files):raise ValueError("Need distinct dynamic data files")
    cells,records,ids,seeds={},[],[],[]
    for i,path in enumerate(files):
        with h5py.File(path) as f:
            if f.attrs.get("schema")!="tactile_dynamic_normal_v1" or f.attrs.get("status")!="complete":
                raise ValueError(f"Incomplete or wrong dynamic data: {path}")
            meta=json.loads(f.attrs["metadata_json"])
            require_matching_calibration(ck["tactile_calibration"],meta["tactile_calibration"])
            uid=str(f.attrs["dataset_uuid"])
            if uid in ids or meta["seed"] in seeds:raise ValueError("Repeated dataset UUID or simulation seed")
            ids.append(uid);seeds.append(meta["seed"])
            valid=f["quality/sample_valid"][:];grasps=f["ids/grasp_id"][:];angles=f["target/in_hand_angle_deg"][:]
            for group in np.unique(grasps[valid]):
                rows=np.flatnonzero(valid&(grasps==group))
                bin_id=int(np.clip(np.floor((angles[rows].mean()+15)/6),0,4))
                group_records=[{"file":i,"row":int(row),"group":f"{uid}:{group}","object_id":meta["object_id"]} for row in rows]
                cells.setdefault((meta["object_id"],bin_id),[]).append(group_records)
                records.extend(group_records)
    result={"files":list(map(str,files)),"dataset_uuids":ids,"simulation_seeds":seeds,
            "split_protocol":"Independent physical grasp groups, stratified by object and initial angle bin"}
    if external:
        previous=ck["dynamic_manifest"]
        if set(previous["dataset_uuids"])&set(ids) or set(previous["simulation_seeds"])&set(seeds):
            raise ValueError("External test must use new collection UUIDs and seeds")
        result["splits"]={"external":records};return result
    rng=np.random.default_rng(seed)
    splits={key:[] for key in ("train","validation","test")}
    for cell,groups in sorted(cells.items()):
        if len(groups)<4:raise ValueError(f"Too few independent valid grasps in {cell}: {len(groups)}")
        rng.shuffle(groups);n=max(1,round(.15*len(groups)))
        for name,chosen in (("validation",groups[:n]),("test",groups[n:2*n]),("train",groups[2*n:])):
            for group in chosen:splits[name].extend(group)
    result["splits"]=splits
    for a,b in (("train","test"),("train","validation"),("validation","test")):
        assert not {r["group"] for r in splits[a]}&{r["group"] for r in splits[b]}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,nargs="+",required=True)
    p.add_argument("--checkpoint",type=Path,default=Path("outputs/normal_angle_v4_model_20261012/best.pt"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--batch_size",type=int,default=64)
    p.add_argument("--learning_rate",type=float,default=2e-4)
    p.add_argument("--seed",type=int,default=10)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--evaluate_only",action="store_true")
    args=p.parse_args()
    if min(args.epochs,args.batch_size)<1 or args.learning_rate<=0:p.error("Invalid training parameters")
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if ck["schema"]!=CHECKPOINT_SCHEMA:raise ValueError("Expected scalar normal-angle network")
    manifest=manifest_for(args.data,ck,args.seed,args.evaluate_only)
    (args.output/"splits.json").write_text(json.dumps(manifest,indent=2)+"\n")
    datasets={key:DynamicDataset(manifest,key,ck["config"]["image_size_hw"]) for key in manifest["splits"]}
    loaders={key:DataLoader(ds,batch_size=args.batch_size,shuffle=key=="train") for key,ds in datasets.items()}
    model=NormalAngleNet(**ck["config"]["model"]).to(args.device);model.load_state_dict(ck["model_state"])
    baseline=ck["baseline_train_mean_deg"] if args.evaluate_only else float(np.mean([x["angle"].item() for x in datasets["train"].cache]))
    if not args.evaluate_only:
        best=float("inf");optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-5)
        with (args.output/"history.jsonl").open("x") as journal:
            for epoch in range(1,args.epochs+1):
                model.train();loss_sum=count=0
                for batch in loaders["train"]:
                    pred=model(batch["left"].to(args.device),batch["right"].to(args.device),batch["opening"].to(args.device))
                    loss=((pred-batch["angle"].to(args.device))/15).square().mean()
                    if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                    loss_sum+=float(loss.detach())*len(pred);count+=len(pred)
                val,_=evaluate_angles(model,loaders["validation"],args.device,baseline)
                row={"epoch":epoch,"train_loss":loss_sum/count,"validation":val}
                journal.write(json.dumps(row)+"\n");journal.flush()
                if val["mae_deg"]<best:
                    best=val["mae_deg"]
                    chosen={**ck,"epoch":epoch,"model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        "baseline_train_mean_deg":baseline,"validation_metrics":val,"dynamic_manifest":manifest,
                        "dynamic_adaptation":{"source_checkpoint":str(args.checkpoint.resolve()),"epochs":args.epochs,
                            "seed":args.seed,"learning_rate":args.learning_rate,"objects":sorted({r['object_id'] for r in manifest['splits']['train']}),
                            "dataset_schema":"tactile_dynamic_normal_v1","allowed_off_axis_residual_deg":3.,
                            "target":"same signed TCP-Y angle; physical grasps can have small measured off-axis residuals"}}
                    torch.save(chosen,args.output/"best.pt")
                scheduler.step()
                print(f"DYNAMIC_ADAPT {epoch}/{args.epochs} val_MAE={val['mae_deg']:.3f} best={best:.3f}",flush=True)
        ck=torch.load(args.output/"best.pt",map_location="cpu",weights_only=True);model.load_state_dict(ck["model_state"])
    report={"status":"complete","checkpoint":str((args.checkpoint if args.evaluate_only else args.output/'best.pt').resolve()),
            "split_samples":{k:len(v) for k,v in datasets.items()},
            "split_grasps":{k:len({r['group'] for r in records}) for k,records in manifest['splits'].items()}}
    for split,loader in loaders.items():
        metrics,arrays=evaluate_angles(model,loader,args.device,baseline)
        report[split]=metrics;np.savez_compressed(args.output/f"{split}_predictions.npz",**arrays)
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print("DYNAMIC_ADAPT_REPORT "+json.dumps(report),flush=True)


if __name__=="__main__":main()
