"""Qualitative examples: question, gold answer and each model's saved prediction, side by side.

    python qualitative.py ce0.3_L7 teacher_chat sft_teacher_L7
    python qualitative.py ce0.3_L7 teacher_chat --n 10 --seed 1 --out results/qualitative.md

Uses the predictions already saved by evaluate.py (no GPU, no generation).
Examples are sampled at random from the test set (fixed seed), so they are not cherry-picked.
"""
import argparse
import json
import random


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="eval/<run> names; the first one is the model being inspected")
ap.add_argument("--dataset", default="pubmedqa")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="results/qualitative.md")
args = ap.parse_args()

tests = {r["id"]: r for r in read_jsonl(f"data/splits/{args.dataset}/test.jsonl")}
preds = {run: {p["id"]: p for p in read_jsonl(f"eval/{run}/{args.dataset}/predictions.jsonl")} for run in args.runs}
ids = sorted(set.intersection(*(set(p) for p in preds.values())))
ids = random.Random(args.seed).sample(ids, min(args.n, len(ids)))

lines = [f"# Qualitative examples ({args.dataset}, {len(ids)} random test cases, seed {args.seed})\n"]
for k, i in enumerate(ids, 1):
    r = tests[i]
    lines += [f"## {k}. `{i}`\n", "**Input**\n", "```", r["prompt"], "```\n",
              f"**Gold:** {r['gold']}. {r.get('long_answer', '')}\n"]
    for run in args.runs:
        p = preds[run][i]
        mark = "" if "acc" not in p else (" ✓" if p["acc"] else " ✗")
        lines += [f"**{run}** (label: {p.get('label')}{mark})\n", "> " + p["prediction"].strip().replace("\n", "\n> ") + "\n"]
    lines.append("---\n")

open(args.out, "w").write("\n".join(lines))
print(f"wrote {args.out}")
for run in args.runs:
    print(f"{run}: {sum(preds[run][i].get('acc', 0) for i in ids):.0f}/{len(ids)} correct in this sample")