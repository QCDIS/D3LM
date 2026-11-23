"""Energy per query, one-time energy and break-even for every evaluated model (after run_energy.sh).

    python energy_table.py                 # all 3 datasets
    python energy_table.py climate         # one dataset
    python energy_table.py --all           # also the superseded 20k-data PubMedQA runs

Per model:
  quality    acc (PubMedQA), BERTScore vs. the teacher (climate), pass@1 (code)
  J/query    bench/<dataset>__<architecture>.json, batch 32, fixed output length (run_energy.sh)
  saving     1 - J/query / teacher J/query
  once (kWh) teacher data generation (data/teacher*/<tag>/manifest.json) + training of every stage (runs/<run>/summary.json)
  N*         break-even = once / (teacher J/query - J/query) queries; "never" if not cheaper than the teacher,
             "0" if nothing is trained (quantization, zero-shot)
The data-generation energy is counted in full for every model trained on the teacher's answers (each model is
treated as if it were the only one built). Hyper-parameter search is not included.
Writes results/energy_table.csv. Missing inputs are listed at the end.
"""
import csv
import json
import os
import re
import sys

KWH = 3.6e6
DATASETS = {   # dataset: metric, teacher eval run, teacher-answers manifest per run prefix
    "pubmedqa": ("acc", "teacher_chat", {"pubmedqa-full": "data/teacher_full/llama31-8b-it-chat/manifest.json",
                                          "pubmedqa": "data/teacher/llama31-8b-it-chat/manifest.json"}),
    "climate": ("fid_bertscore", "climate_teacher", {"climate": "data/teacher/llama31-8b-it-chat/manifest.json"}),
    "code": ("pass", "code_teacher", {"code": "data/teacher/qwen25-coder-7b-it-chat/manifest.json"}),
}
TEACHERS = ("llama31-8b", "qwen25coder-7b")
SMALL = {"llama32-1b": "llama32-1b", "Llama-3.2-1B-Instruct": "llama32-1b", "qwen25coder-1.5b": "qwen25coder-1.5b",
         "qwen25-1.5b": "qwen25-1.5b", "Qwen2.5-1.5B-Instruct": "qwen25-1.5b", "smollm2-1.7b": "smollm2-1.7b",
         "SmolLM2-1.7B-Instruct": "smollm2-1.7b", "gemma3-1b": "gemma3-1b", "gemma-3-1b-it": "gemma3-1b"}
missing = set()


def load(path):
    if os.path.exists(path):
        return json.load(open(path))
    missing.add(path)
    return None


def j_per_query(ds, arch):
    b = load(f"bench/{ds}__{arch}.json")
    if b is None:
        return None
    r = next((r for r in b["hf"] if r["batch_size"] == 32), b["hf"][-1])
    return r["mean"]["j_per_request"]


def parse(run):
    """-> (architecture, training runs of all stages, trained on the teacher's answers?) or None if not a model row."""
    prefix, model, rest = (run.split("_", 2) + ["", ""])[:3]
    if rest in ("int8", "nf4"):
        return f"teacher_{rest}", [], False
    if rest == "zs":
        return SMALL.get(model), [], False
    if not re.search(r"_\d+st$", run):
        return None
    steps = run.rsplit("_", 1)[1]
    depths = re.findall(r"-p(\d+)", model)
    if not depths:                                   # small pretrained student
        return SMALL.get(model), [run], "gold" not in rest
    base = model.split("-p")[0]
    stages = [f"{prefix}_{base}{''.join('-p' + d for d in depths[:k])}_{'ce' if k == 1 else 'kd'}_{steps}"
              for k in range(1, len(depths))]        # earlier stages of gradual pruning (32 -> 14 -> 7 -> 5)
    chain = base + "".join("-p" + d for d in depths)
    if model != chain:                               # continued from the full chain (e.g. -l25: only some layers tuned)
        stages.append(f"{prefix}_{chain}_{'ce' if len(depths) == 1 else 'kd'}_{steps}")
    return f"p{depths[-1]}", stages + [run], "gold" not in rest


def rows_for(ds, show_all):
    metric, teacher_run, manifests = DATASETS[ds]
    e_t = j_per_query(ds, "teacher")
    out = []
    runs = sorted(os.listdir("eval")) if os.path.isdir("eval") else []
    for run in [teacher_run] + [r for r in runs if r != teacher_run]:
        m = load(f"eval/{run}/{ds}/metrics.json") if os.path.exists(f"eval/{run}/{ds}") or run == teacher_run else None
        if m is None:
            continue
        if ds == "pubmedqa" and run.startswith("pubmedqa_") and run.endswith("st") and not show_all:
            continue                                 # 20k subset, superseded by pubmedqa-full_*
        p = ("teacher", [], False) if run == teacher_run else parse(run)
        if p is None or p[0] is None:
            continue
        arch, stages, teacher_data = p
        q = m.get(metric)
        q = q["mean"] if isinstance(q, dict) else q
        if q is None and run == teacher_run:
            q = 1.0                                  # climate: the teacher is the reference
        e = j_per_query(ds, arch)
        once = 0.0
        if stages:
            sums = [load(f"runs/{s}/summary.json") for s in stages]
            once = None if None in sums else sum(s["energy"]["energy_j"] for s in sums)
            if once is not None and teacher_data:
                man = load(manifests[run.split("_")[0]]) or {}
                once += sum(man[k]["energy"]["energy_j"] for k in (f"{ds}_train", f"{ds}_validation") if k in man)
        if run == teacher_run:
            nstar = "-"
        elif e is None or e_t is None or once is None:
            nstar = "?"
        elif e >= e_t:
            nstar = "never"
        else:
            nstar = f"{once / (e_t - e):,.0f}"
        out.append({"dataset": ds, "run": run, "architecture": arch, "quality": q, "j_per_query": e,
                    "saving": None if e is None or e_t is None else 1 - e / e_t,
                    "once_kwh": None if once is None else once / KWH, "break_even_queries": nstar})
    return out


def fmt(v, f):
    return "-" if v is None else format(v, f)


show_all = "--all" in sys.argv
all_rows = []
for ds in [a for a in sys.argv[1:] if a != "--all"] or list(DATASETS):
    rows = rows_for(ds, show_all)
    all_rows += rows
    print(f"\n=== {ds} ({DATASETS[ds][0]})")
    print(f"{'run':48}{'arch':>17}{'quality':>9}{'J/query':>9}{'saving':>8}{'once kWh':>10}{'N*':>14}")
    for r in rows:
        print(f"{r['run'][:48]:48}{r['architecture']:>17}{fmt(r['quality'], '.3f'):>9}{fmt(r['j_per_query'], '.1f'):>9}"
              f"{fmt(r['saving'], '.0%'):>8}{fmt(r['once_kwh'], '.2f'):>10}{r['break_even_queries']:>14}")
os.makedirs("results", exist_ok=True)
if all_rows:
    with open("results/energy_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print("\nwrote results/energy_table.csv")
if missing:
    print("missing:\n  " + "\n  ".join(sorted(missing)))