"""Introduction figure (single column): energy per query and accuracy of each compression route, one row each.

    python intro_figure.py                         # PubMedQA -> results/intro_figure.pdf / .png
    python intro_figure.py --dataset climate       # results/intro_figure_climate.pdf / .png

Left panel: GPU energy per query (J, HF generate at batch 32, fixed output length), with the number of queries after
which a trained model has repaid its one-time energy (teacher data generation + training) against the teacher.
Right panel: the dataset's main quality metric; the dashed line is the teacher.
Inputs (missing ones are reported, the row or value is left out): eval/<run>/<dataset>/metrics.json,
bench/*.json (COMMANDS.md sections 8 and 30), runs/<run>/summary.json, data/teacher/<tag>/manifest.json.
PubMedQA rows: teacher (BF16, INT8, NF4) and the pruned teacher with 14 and with 7 layers. The 14-layer energy needs
bench/pubmedqa_p14.json (COMMANDS.md section 30, last block; run with nothing else on the GPU).
"""
import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# dataset: metric shown, its axis label, teacher answers manifest (for the data-generation energy)
DATASETS = {
    "pubmedqa": ("acc", "Accuracy (%)", "data/teacher/llama31-8b-it-chat/manifest.json"),
    "climate": ("fid_bertscore", "BERTScore vs. teacher (%)", "data/teacher/llama31-8b-it-chat/manifest.json"),
    "code": ("pass", "pass@1 (%)", "data/teacher/qwen25-coder-7b-it-chat/manifest.json"),
}
# rows, top to bottom: (label, eval run or None = teacher reference, bench file,
#                       training run(s) whose energy counts for the break-even (a list for multi-stage models), style)
ROWS = {
    "pubmedqa": [
        ("Teacher, Llama-3.1-8B (BF16)", "teacher_chat", "bench/teacher.json", None, "teacher"),
        ("Teacher, INT8", "pubmedqa_llama31-8b_int8", "bench/teacher_int8.json", None, "int8"),
        ("Teacher, NF4", "pubmedqa_llama31-8b_nf4", "bench/teacher_nf4.json", None, "nf4"),
        # pruned models at 3000 steps (same budget for both); gradual = 32 -> 14 -> 7 layers;
        # bench/student.json = a 7-layer model (section 8)
        ("Pruned to 7 layers, gradual", "pubmedqa_llama31-8b-p14-p7_kd_3000st", "bench/student.json",
         ["pubmedqa_llama31-8b-p14_ce_1500st", "pubmedqa_llama31-8b-p14-p7_kd_3000st"], "prune"),
        ("Pruned to 7 layers, one step", "pubmedqa_llama31-8b-p7_kdhid_3000st", "bench/student.json",
         "pubmedqa_llama31-8b-p7_kdhid_3000st", "prune2"),
    ],
    "climate": [
        ("Teacher, Llama-3.1-8B (BF16)", None, "bench/climate_teacher.json", None, "teacher"),
        ("Teacher, INT8", "climate_llama31-8b_int8", "bench/climate_teacher_int8.json", None, "int8"),
        ("Teacher, NF4", "climate_llama31-8b_nf4", "bench/climate_teacher_nf4.json", None, "nf4"),
        ("Pruned to 7 layers, gradual", "climate_llama31-8b-p14-p7_kd_1500st", "bench/climate_p7.json",
         ["climate_llama31-8b-p14_ce_1500st", "climate_llama31-8b-p14-p7_kd_1500st"], "prune"),
        ("Pruned to 7 layers, one step", "climate_llama31-8b-p7_kdhid_1500st", "bench/climate_p7.json",
         "climate_llama31-8b-p7_kdhid_1500st", "prune2"),
    ],
    "code": [
        ("Teacher, Qwen2.5-Coder-7B (BF16)", "code_teacher", "bench/code_teacher.json", None, "teacher"),
        ("Teacher, INT8", "code_qwen25coder-7b_int8", "bench/code_teacher_int8.json", None, "int8"),
        ("Teacher, NF4", "code_qwen25coder-7b_nf4", "bench/code_teacher_nf4.json", None, "nf4"),
        ("KD into Qwen2.5-Coder-1.5B", "code_qwen25coder-1.5b_kd_1500st", "bench/code_small.json",
         "code_qwen25coder-1.5b_kd_1500st", "kd"),
        ("Pruned to 7 of 28 layers", "code_qwen25coder-7b-p7_kdhid_1500st", "bench/code_p7.json",
         "code_qwen25coder-7b-p7_kdhid_1500st", "prune"),
    ],
}
# fill, edge, hatch per style; a second variant of the same route (NF4, one-step pruning) is hatched
STYLE = {"teacher": ("#4a4a48", "#4a4a48", None), "int8": ("#2a78d6", "#2a78d6", None),
         "nf4": ("white", "#2a78d6", "/////"), "kd": ("#eb6834", "#eb6834", None),
         "prune": ("#1baf7a", "#1baf7a", None), "prune2": ("white", "#1baf7a", "/////")}
INK, GREY = "#1a1a1a", "#6b6b68"
missing = []


def load(path):
    if path and os.path.exists(path):
        return json.load(open(path))
    if path:
        missing.append(path)
    return None


def metric(ds, run):
    if run is None:            # the teacher is the reference (climate: every metric is measured against it)
        return 1.0
    m = load(f"eval/{run}/{ds}/metrics.json")
    v = None if m is None else m.get(DATASETS[ds][0])
    return v["mean"] if isinstance(v, dict) else v


def energy(bench):
    b = load(bench)
    if b is None:
        return None
    r = next((r for r in b.get("hf", []) if r["batch_size"] == 32), None) or b["hf"][-1]
    return r["mean"]["j_per_request"]


def one_time_j(ds, runs):
    """Teacher data generation + training energy of every stage (runs: a run name or a list of them)."""
    summaries = [load(f"runs/{r}/summary.json") for r in ([runs] if isinstance(runs, str) else runs)]
    if None in summaries:
        return None
    man = load(DATASETS[ds][2]) or {}
    gen = sum(man[k]["energy"]["energy_j"] for k in (f"{ds}_train", f"{ds}_validation") if k in man)
    return gen + sum(s["energy"]["energy_j"] for s in summaries)


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", default="pubmedqa", choices=list(DATASETS))
ap.add_argument("--out", default=None)
args = ap.parse_args()
ds = args.dataset
out = args.out or ("results/intro_figure" if ds == "pubmedqa" else f"results/intro_figure_{ds}")

rows = []
for label, run, bench, train, style in ROWS[ds]:
    q, e = metric(ds, run), energy(bench)
    if q is None:
        continue
    rows.append({"label": label, "q": 100 * q, "e": e, "train": train, "style": style})
e_teacher = next((r["e"] for r in rows if r["style"] == "teacher"), None)
q_teacher = next((r["q"] for r in rows if r["style"] == "teacher"), None)
for r in rows:   # queries after which the one-time energy is repaid by the lower energy per query
    r["payback"] = None
    if r["train"] and r["e"] and e_teacher and r["e"] < e_teacher:
        once = one_time_j(ds, r["train"])
        r["payback"] = None if once is None else once / (e_teacher - r["e"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif",
                                           "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.5, "axes.edgecolor": "#444444", "xtick.major.width": 0.5, "xtick.major.size": 2,
    "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})

n = len(rows)
fig, (ax_e, ax_q) = plt.subplots(1, 2, figsize=(3.5, 0.32 * n + 0.55), sharey=True,
                                 gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.12})
ys = list(range(n))[::-1]
e_max = max((r["e"] for r in rows if r["e"]), default=1)
for y, r in zip(ys, rows):
    face, edge, hatch = STYLE[r["style"]]
    bar = dict(height=0.62, color=face, edgecolor=edge, hatch=hatch, linewidth=0.7)
    if r["e"]:
        ax_e.barh(y, r["e"], **bar)
        text = f"{r['e']:.0f} J"
        if r["payback"]:
            text += f"; break-even at {r['payback'] / 1e3:.0f}k queries"
        ax_e.text(r["e"] + 0.02 * e_max, y, text, va="center", fontsize=6.6, color=INK)
    else:
        ax_e.text(0.02 * e_max, y, "not measured", va="center", fontsize=6.6, color=GREY, style="italic")
    ax_q.barh(y, r["q"], **bar)
    light = face == "white"   # value inside the bar end; on the hatched bar on a white patch
    ax_q.text(r["q"] - 2, y, f"{r['q']:.1f}", va="center", ha="right", fontsize=6.6,
              color=edge if light else "white", bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none") if light else None)

ax_e.set_yticks(ys, [r["label"] for r in rows])
ax_e.set_xlim(0, e_max * 1.9)
ax_e.set_xlabel("Energy per query (J)")
ax_q.set_xlim(0, 100)
ax_q.set_xticks([0, 25, 50, 75, 100])
ax_q.set_xlabel(DATASETS[ds][1])
if q_teacher:
    ax_q.axvline(q_teacher, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
for ax in (ax_e, ax_q):
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, n - 0.4)

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.02, dpi=300)
print(f"wrote {out}.pdf and {out}.png")
for r in rows:
    pb = f", break-even at {r['payback']:.0f} queries" if r["payback"] else ""
    print(f"  {r['label']:34} {DATASETS[ds][0]} {r['q']:.1f}%  energy {r['e'] or float('nan'):.1f} J{pb}")
if missing:
    print("missing (left out):\n  " + "\n  ".join(sorted(set(missing))))