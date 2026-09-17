#!/usr/bin/env python3
import re
from pathlib import Path
from datetime import datetime

def parse(log_path):
    s = Path(log_path).read_text(errors="ignore")
    gens = [float(x) for x in re.findall(r"vLLM generation done - elapsed time: ([0-9.]+)s", s)]
    n = 32
    step_means = []
    step_maxs = []
    for i in range(0, len(gens)//n*n, n):
        chunk = gens[i:i+n]
        step_means.append(sum(chunk)/len(chunk))
        step_maxs.append(max(chunk))
    ts = re.findall(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", s)
    print("FILE", log_path)
    print(" n_gen", len(gens), "n_steps", len(step_means))
    if step_means:
        print(" microbatch gen s: min/mean/max", round(min(step_means),2), round(sum(step_means)/len(step_means),2), round(max(step_means),2))
        wall = [8*x for x in step_means]  # 8 grad accum sequential; 4 ranks parallel
        print(" EST wall gen/step s (8*mean): min/mean/max", round(min(wall),1), round(sum(wall)/len(wall),1), round(max(wall),1))
        print(" EST total gen hours", round(sum(wall)/3600, 2))
        # first 5 vs last 5
        print(" first5 wall", [round(x,1) for x in wall[:5]], "last5", [round(x,1) for x in wall[-5:]])
        print(" step-to-step cv", round((max(wall)-min(wall))/ (sum(wall)/len(wall)), 3))
    print(" timestamps", len(ts), ts[:1], ts[-1:] if ts else None)
    # HF {'loss':
    steps = re.findall(r"'step': (\d+)", s)
    print(" quoted steps", steps[:5], steps[-5:] if steps else None)
    p = Path(log_path)
    print(" mtime", datetime.fromtimestamp(p.stat().st_mtime).isoformat())
    print("---HEAD---")
    print(s[:600])
    print("---TAIL---")
    print(s[-900:])
    print("====\n")

parse("/home1/cjl/ICLR2026/opsd_100steps_retry2.log")
parse("/home1/cjl/ICLR2026/qwen34b_100steps.log")
