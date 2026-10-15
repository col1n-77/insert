"""Run two equal-budget training arms sequentially, then evaluate frozen weights.

This process starts work immediately; it is not a scheduled automation.
It never reuses an existing experiment directory or overwrites checkpoints.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--study_name")
p.add_argument("--task", default="InsertAnythingAngle-LHole-III-Direct-v0")
p.add_argument("--max_iterations", type=int, default=1000)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--angle_noise_std_deg", type=float, default=0.)
p.add_argument("--eval_seeds", type=int, nargs="+", default=[1001,2002])
p.add_argument("--learning_rate", type=float, default=1.e-4)
p.add_argument("--only", choices=["aligned","control"], help="Run one arm instead of the paired study")
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
name = args.study_name or datetime.now().strftime("LHole_III_reward_ablation_%Y%m%d_%H%M%S")
if Path(name).name != name: p.error("study_name must be a directory name")
if args.max_iterations < 1 or args.num_envs < 1: p.error("Positive iterations and num_envs required")
if args.seed in args.eval_seeds or len(set(args.eval_seeds)) != len(args.eval_seeds):
    p.error("Evaluation seeds must be distinct and differ from training")
out = root/"outputs"/name
out.mkdir(parents=True, exist_ok=False)
state = {"schema":"insertanything_angle_reward_study_v2", "status":"running",
         "axis_reward_form":"gaussian_minus_one",
         "started_at":datetime.now().astimezone().isoformat(), "pid":os.getpid(),
         "arguments":vars(args), "arms":{}, "comparison_note":"same observations/seed/budget; weight1 versus weight0. Single training seed, descriptive evidence only."}
base_task = Path(os.environ.get("INSERTANYTHING_ROOT", str(root.parent)))/"source/InsertAnything/InsertAnything/tasks/direct/insertanything"
tracked = [*root.joinpath("source").rglob("*.py"), *root.joinpath("scripts").glob("*.py"),
           *base_task.glob("*.py"), *base_task.joinpath("agents").glob("*.yaml")]
expected_sources = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(tracked)}
state["source_sha256"] = expected_sources

def save():
    target = out/"study.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2)+"\n")
    temp.replace(target)

def execute(label, command, logfile):
    for path, expected in expected_sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed during paired experiment: {path}. Preserve this study and use a fresh run after review.")
    print(f"STUDY_START {label}: {' '.join(command)}", flush=True)
    with logfile.open("w") as stream:
        child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        state.update(current_stage=label, child_pid=child.pid, current_log=str(logfile))
        save()
        code = child.wait()
    if code: raise RuntimeError(f"{label} exited {code}; see {logfile}")
    state["child_pid"] = None
    save()
    print(f"STUDY_DONE {label}", flush=True)

save()
try:
    for label, weight in [("aligned",1.),("control",0.)]:
        if args.only and args.only != label: continue
        run_name = f"{name}_{label}"
        run = root/"logs/rl_games/InsertAnythingAngle"/run_name
        arm = state["arms"][label] = {"weight":weight,"training_run":str(run),"status":"training"}
        command = [str(root/"scripts/python.sh"),"scripts/train_angle_rl.py","--headless",
                   "--task",args.task,"--num_envs",str(args.num_envs),"--seed",str(args.seed),
                   "--max_iterations",str(args.max_iterations),"--angle_noise_std_deg",str(args.angle_noise_std_deg),
                   "--align_weight",str(weight),"--learning_rate",str(args.learning_rate),"--run_name",run_name]
        execute(f"train_{label}", command, out/f"train_{label}.log")
        contract = json.loads((run/"run.json").read_text())
        if contract["status"] != "completed": raise RuntimeError("Trainer did not mark run completed")
        candidates = sorted((run/"nn").glob(f"last_*_ep_{args.max_iterations}_*.pth"))
        if not candidates: raise RuntimeError(f"No final epoch {args.max_iterations} checkpoint in {run}")
        final = candidates[0]
        selected = {"final":final}
        best = run/"nn/InsertAnythingAngle.pth"
        if best.is_file(): selected["best_training_reward"] = best
        arm.update(status="evaluating", evaluations={})
        for selection, checkpoint in selected.items():
            output = out/f"eval_{label}_{selection}"
            command = [str(root/"scripts/python.sh"),"scripts/eval_angle_policy.py","--headless",
                       "--task",args.task,"--checkpoint",str(checkpoint),"--training_run",str(run),
                       "--num_envs",str(args.num_envs),"--seeds",*map(str,args.eval_seeds),
                       "--modes","deterministic","stochastic","--output",str(output)]
            execute(f"eval_{label}_{selection}", command, out/f"eval_{label}_{selection}.log")
            report = json.loads((output/"report.json").read_text())
            if report["status"] != "completed": raise RuntimeError("Evaluation incomplete")
            arm["evaluations"][selection] = {"report":str(output/"report.json"),"epoch":report["checkpoint_epoch"]}
            save()
        arm["status"] = "completed"
        save()
    state.update(status="summarizing",current_stage="summarizing")
    save()
    command = [str(root/"scripts/python.sh"),"scripts/summarize_reward_study.py","--study",str(out)]
    execute("summarize",command,out/"summarize.log")
    state.update(status="completed",current_stage="completed",finished_at=datetime.now().astimezone().isoformat())
    save()
    print("ANGLE_STUDY_OUTPUT "+str(out), flush=True)
except BaseException as exc:
    state.update(status="failed",error=repr(exc))
    save()
    traceback.print_exc()
    sys.exit(1)
