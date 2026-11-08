"""Train + evaluate a few variants and collect the results in one CSV.

    python hparam_sweep.py configs/pubmedqa_llama_L7.yaml
    python hparam_sweep.py configs/pubmedqa_llama_L7.yaml clean clean_ce05     # only some variants

Results: results/hparam_sweep.csv (one row per variant). Variants already evaluated are skipped.
"""
import csv
import json
import os
import re
import subprocess
import sys

VARIANTS = {
    "base":            [],
    "clean":           ["teacher_tag=llama31-8b-it-clean"],
    "clean_ce05":      ["teacher_tag=llama31-8b-it-clean", "train.ce_weight=0.5"],
    "clean_T1":        ["teacher_tag=llama31-8b-it-clean", "train.temperature=1.0"],
    "clean_rkl":       ["teacher_tag=llama31-8b-it-clean", "train.divergence=rkl"],
    "clean_skl":       ["teacher_tag=llama31-8b-it-clean", "train.divergence=skl"],
    "clean_edge":      ["teacher_tag=llama31-8b-it-clean", "student.layer_ids=[0,1,2,3,29,30,31]"],
}


def clean_targets(src="data/teacher/llama31-8b-it", dst="data/teacher/llama31-8b-it-clean"):
    """Keep only the label and the first sentence of each teacher answer."""
    os.makedirs(dst, exist_ok=True)
    for split in ("train", "validation"):
        out = f"{dst}/pubmedqa_{split}.jsonl"
        if os.path.exists(out):
            continue
        with open(f"{src}/pubmedqa_{split}.jsonl") as fin, open(out, "w") as fout:
            for line in fin:
                r = json.loads(line)
                t = re.split(r"\(Note", r["completion"].split("\n\n")[0])[0].strip()
                m = re.match(r"\W*(yes|no|maybe)\b\W*(.*)", t, re.I | re.S)
                if m:
                    first = re.split(r"(?<=[.!?])\s", m.group(2), maxsplit=1)[0]
                    t = f"{m.group(1).capitalize()}. {first}".strip()
                fout.write(json.dumps({**r, "completion": " " + t}) + "\n")


cfg = sys.argv[1]
names = sys.argv[2:] or list(VARIANTS)
clean_targets()
os.makedirs("results", exist_ok=True)
for name in names:
    run = f"hp_{name}"
    metrics_file = f"eval/{run}/pubmedqa/metrics.json"
    if os.path.exists(metrics_file):
        print(f"{name}: already done, skipping")
        continue
    sets = VARIANTS[name] + [f"run_name={run}"]
    subprocess.run(["python", "train.py", "--config", cfg, "--set", *sets])
    subprocess.run(["python", "evaluate.py", "--config", cfg, "--set", *sets, "--name", run,
                    "--model", f"runs/{run}/best", "--reference", "teacher"])
    if not os.path.exists(metrics_file):
        print(f"{name}: failed, see the output above")
        continue
    m = json.load(open(metrics_file))
    s = json.load(open(f"runs/{run}/summary.json"))
    c = json.load(open(f"runs/{run}/config.json"))
    row = {"variant": name, "settings": " ".join(VARIANTS[name]),
           "layers": c["student"]["num_layers"], "lr": c["train"]["lr"], "ce_weight": c["train"]["ce_weight"],
           "temperature": c["train"]["temperature"], "divergence": c["train"]["divergence"],
           "freeze": c["student"]["freeze"], "best_val_ce": s["best_val_ce"], "steps": s["steps"],
           "train_kwh": s["energy"].get("kwh"), "acc": m["acc"]["mean"], "macro_f1": m["macro_f1"],
           "gold_rougeL": m["gold_rougeL"]["mean"], "gold_bertscore": m.get("gold_bertscore", {}).get("mean"),
           "fid_rougeL": m["fid_rougeL"]["mean"], "fid_ppl": m.get("fid_ppl", {}).get("mean"),
           "stop_rate": m["failure"]["stop_rate"]}
    new = not os.path.exists("results/hparam_sweep.csv")
    with open("results/hparam_sweep.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(row)
    