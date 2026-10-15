"""Summarize paired success/axis metrics without comparing incompatible returns."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--study",type=Path,required=True)
a=p.parse_args()
state=json.loads((a.study/"study.json").read_text())

def wilson(k,n):
    z=1.959963984540054
    center=(k/n+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-radius,center+radius]

complete = bool(state["arms"]) and all(arm.get("status") == "completed" for arm in state["arms"].values())
summary={"study":str(a.study.resolve()),"status":"completed" if complete else state["status"],"arms":{},
         "primary_comparison":"final checkpoint at equal training epochs; strict success and terminal axis error",
         "limitations":"One training seed; episode confidence intervals exclude training-seed variation. Best-training-reward selection is secondary. Returns across reward weights are not a comparable success metric."}
for label,arm in state["arms"].items():
    selections={}
    for selection,ev in arm.get("evaluations",{}).items():
        report=json.loads(Path(ev["report"]).read_text())
        modes={}
        for mode,result in report["results"].items():
            ep=result["episodes"];n=len(ep)
            metrics={"episodes":n}
            for key in ("native_success","strict_success","terminal_native_success","terminal_strict_success"):
                k=sum(bool(e[key]) for e in ep)
                metrics[key]={"count":k,"rate":k/n,"wilson_95":wilson(k,n)}
            axis=[e["terminal_axis_error_deg"] for e in ep]
            metrics.update(terminal_axis_mean_deg=float(np.mean(axis)),terminal_axis_median_deg=float(np.median(axis)),
                           terminal_axis_p95_deg=float(np.percentile(axis,95)))
            modes[mode]=metrics
        selections[selection]={"epoch":report["checkpoint_epoch"],"report":ev["report"],"modes":modes}
    summary["arms"][label]=selections
if all("final" in summary["arms"].get(label,{}) for label in ("aligned","control")):
    summary["aligned_minus_control"]={}
    for mode in ("deterministic","stochastic"):
        x=summary["arms"]["aligned"]["final"]["modes"][mode]
        y=summary["arms"]["control"]["final"]["modes"][mode]
        summary["aligned_minus_control"][mode]={"strict_success_percentage_points":100*(x["strict_success"]["rate"]-y["strict_success"]["rate"]),
             "terminal_strict_success_percentage_points":100*(x["terminal_strict_success"]["rate"]-y["terminal_strict_success"]["rate"]),
             "terminal_axis_mean_deg":x["terminal_axis_mean_deg"]-y["terminal_axis_mean_deg"]}
(a.study/"comparison.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
print(json.dumps(summary,indent=2,allow_nan=False))
