"""Results of the comparison runs (run_internals.sh), one table per dataset.

    python print_results.py              # all datasets
    python print_results.py climate      # one dataset
"""
import glob
import json
import os
import sys

KEYS = {"pubmedqa": ["acc", "macro_f1", "fid_rougeL", "fid_ppl"],
        "climate": ["fid_bertscore", "fid_cosine", "fid_rougeL", "fid_ppl"],
        "code": ["pass", "pass_humaneval", "pass_mbpp", "fid_pass_agreement"]}
TEACHER = {"pubmedqa": "teacher_chat", "climate": "climate_teacher", "code": "code_teacher"}

for ds in sys.argv[1:] or KEYS:
    keys = KEYS[ds]
    print(f"\n=== {ds}")
    print(f"{'run':46}" + "".join(f"{k[:13]:>14}" for k in keys) + f"{'best/steps':>12}{'hours':>7}")
    for f in sorted(glob.glob(f"eval/*/{ds}/metrics.json")):
        run = f.split("/")[1]
        if not (run == TEACHER[ds] or run.endswith(("_1500st", "_6500st", "_zs", "_int8", "_nf4"))):
            continue
        m = json.load(open(f))
        g = lambda k: m[k]["mean"] if isinstance(m.get(k), dict) else m.get(k, float("nan"))
        st, h = "-", float("nan")
        if os.path.exists(f"runs/{run}/summary.json"):
            d = json.load(open(f"runs/{run}/summary.json"))
            st = f"{d['best_step']}/{d['steps']}"
        if os.path.exists(f"runs/{run}/log.jsonl"):
            lines = [json.loads(l) for l in open(f"runs/{run}/log.jsonl") if l.strip()]
            h = lines[-1].get("elapsed_s", float("nan")) / 3600 if lines else h
        print(f"{run:46}" + "".join(f"{g(k):14.3f}" for k in keys).replace("nan", "  -")
              + f"{st:>12}" + f"{h:7.1f}".replace("nan", "  -"))