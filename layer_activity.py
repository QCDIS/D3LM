"""Brain-scan style picture of which layers "light up", for any model (teacher or student) and any dataset.

    python layer_activity.py --config configs/pubmedqa_llama_L7_chat.yaml                      # the config's teacher
    python layer_activity.py --config configs/pubmedqa_llama_L7_chat.yaml --model runs/<run>/best
    python layer_activity.py --config configs/code_qwen_L7.yaml --model Qwen/Qwen2.5-Coder-1.5B-Instruct --chat-template

For every validation prompt and layer: activity = angular distance between the hidden state entering and leaving the
layer (arccos(cos) / pi, at the last prompt token; 0 = no change). Output, named by dataset and model:
    results/layer_activity/<dataset>__<model>.png   (scan per prompt, mean per layer, distance map)
    results/layer_activity/<dataset>__<model>.npy   (prompts x layers)
The chat template is used for the config's teacher (if the config uses it) or with --chat-template; trained students
were trained without it. --importance adds an accuracy-drop curve (results/layer_importance.csv) to the middle panel.
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import dataset_name, device_of, load_model, load_split, load_tokenizer, parse_args

args, cfg = parse_args(__doc__, lambda ap: (
    ap.add_argument("--model", default=None, help="HF id or runs/<run>/best (default: the config's teacher)"),
    ap.add_argument("--chat-template", action="store_true"),
    ap.add_argument("--n", type=int, default=300),
    ap.add_argument("--importance", default=None, help="accuracy-drop CSV from layer_importance.py"),
    ap.add_argument("--out-dir", default="results/layer_activity")))
model_id = args.model or cfg["teacher"]
chat = args.chat_template or (args.model is None and cfg.get("teacher_chat_template", False))
spec = cfg["datasets"][0]
ds = dataset_name(spec)
tag = os.path.basename(os.path.dirname(model_id.rstrip("/"))) if model_id.rstrip("/").endswith("/best") \
    else model_id.rstrip("/").split("/")[-1]
out = os.path.join(args.out_dir, f"{ds}__{tag}.png")

rows = load_split(cfg, spec, "validation")[:args.n]
tok = load_tokenizer(model_id)
model = load_model(model_id, "bf16", device_of(cfg))
texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
         if chat else r["prompt"] for r in rows]

tok.padding_side = tok.truncation_side = "left"   # position -1 = last prompt token, even for long prompts
acts, D = [], 0
with torch.no_grad():
    for i in range(0, len(texts), 8):
        enc = tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg["max_prompt_tokens"], add_special_tokens=not chat).to(model.device)
        h = F.normalize(torch.stack(model(**enc, output_hidden_states=True).hidden_states)[:, :, -1].float(), dim=-1)
        ang = torch.arccos(torch.einsum("ibd,jbd->ijb", h, h).clamp(-1, 1)) / math.pi   # (L+1, L+1, batch)
        acts.append(torch.diagonal(ang, offset=1).cpu())                                # (batch, L): d(l, l+1)
        D = D + ang.sum(-1).cpu()
A = torch.cat(acts).numpy()
D = (D / len(texts)).numpy()
os.makedirs(args.out_dir, exist_ok=True)
np.save(out.replace(".png", ".npy"), A)

# group rows by label for label tasks (PubMedQA); otherwise sort by overall activity
labels = [str(r["gold"]) if r.get("task") == "yes_no_maybe" else "" for r in rows]
order = sorted(range(len(rows)), key=lambda i: (labels[i], -A[i].mean()))
L = A.shape[1]
fig = plt.figure(figsize=(11, 13))
gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.2, 3], width_ratios=[40, 1], hspace=0.35, wspace=0.03)
fig.suptitle(f"{tag}  |  {ds}  |  {len(rows)} validation prompts", fontsize=13, y=0.93)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(A[order], aspect="auto", cmap="inferno", interpolation="nearest")
bounds = [k for k in range(1, len(order)) if labels[order[k]] != labels[order[k - 1]]]
for k in bounds:
    ax.axhline(k - 0.5, color="white", lw=1)
for start in [0] + bounds:
    if labels[order[start]]:
        ax.text(-0.8, start + 2, labels[order[start]], ha="right", va="top", fontsize=9)
ax.set_title("Layer activity per prompt" + (" (grouped by gold label)" if any(labels) else ""))
ax.set_ylabel("prompt")
ax.set_xticks(range(L))
ax.set_xticklabels(range(L), fontsize=7)
fig.colorbar(im, cax=fig.add_subplot(gs[0, 1]), label="angular change")

ax = fig.add_subplot(gs[1, 0])
mean = A.mean(0)
ax.bar(range(L), mean, color=plt.cm.inferno(mean / mean.max() * 0.85), edgecolor="none")
ax.set_xlim(-0.5, L - 0.5)
ax.set_xticks(range(L))
ax.set_xticklabels(range(L), fontsize=7)
ax.set_xlabel("layer")
ax.set_ylabel("mean change")
if args.importance and os.path.exists(args.importance):
    drop = [float(r["acc_drop"]) for r in csv.DictReader(open(args.importance))]
    if len(drop) == L:
        ax2 = ax.twinx()
        ax2.plot(range(L), drop, "o-", color="tab:blue", ms=3, lw=1, label="accuracy drop if removed")
        ax2.set_ylabel("accuracy drop", color="tab:blue")
        ax2.legend(loc="upper center", fontsize=8, frameon=False)

ax = fig.add_subplot(gs[2, 0])
im = ax.imshow(D, cmap="inferno", origin="lower")
ax.set_title("Distance between hidden states i and j (how much the layers in between change it)")
ax.set_xlabel("hidden state j (after layer j-1)")
ax.set_ylabel("hidden state i")
fig.colorbar(im, cax=fig.add_subplot(gs[2, 1]), label="angular distance")

fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out} and {out.replace('.png', '.npy')}")