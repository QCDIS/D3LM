"""How the MLP neurons of one layer fire ("electron density" of a layer), for the most and least active layer.

    python neuron_activity.py --config configs/climate_llama_L7.yaml                                    # teacher
    python neuron_activity.py --config configs/climate_llama_L7.yaml --model runs/<run>/best           # student
    python neuron_activity.py --config configs/climate_llama_L7.yaml --model meta-llama/Llama-3.2-1B-Instruct --chat-template
    python neuron_activity.py --config configs/climate_llama_L7.yaml --compare results/neuron_activity/climate__*.npz

Neuron = one channel of the MLP's hidden layer, act(gate(x)) * up(x), the input of down_proj (14336 per layer for
Llama-3.1-8B, 8192 for Llama-3.2-1B). Measured on every prompt token of the validation prompts (padding and the
first token, an attention sink with huge activations, are left out).
Layers: by default the layer that changes the hidden state most and least (angular distance, as in layer_activity.py);
--layers 0 5 picks them by hand.
Per neuron: mean |activation|, and firing rate = share of tokens with |activation| > tau (tau = RMS of that layer's
activations on the first batch). Per layer: dead neurons (fire < 0.1%), share of the total |activation| carried by
the top 1% of neurons, and the participation ratio PR = (sum m)^2 / (N sum m^2) (1 = all neurons equally busy,
small = a few neurons do the work).
Output: results/neuron_activity/<dataset>__<model>.png and .npz; --compare overlays several .npz files.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LogNorm

from common import dataset_name, device_of, load_model, load_split, load_tokenizer, parse_args

args, cfg = parse_args(__doc__, lambda ap: (
    ap.add_argument("--model", default=None, help="HF id or runs/<run>/best (default: the config's teacher)"),
    ap.add_argument("--chat-template", action="store_true"),
    ap.add_argument("--n", type=int, default=300),
    ap.add_argument("--layers", type=int, nargs=2, default=None, metavar=("MOST", "LEAST")),
    ap.add_argument("--compare", nargs="+", default=None, help=".npz files written by this script"),
    ap.add_argument("--out-dir", default="results/neuron_activity")))
ds = dataset_name(cfg["datasets"][0])
os.makedirs(args.out_dir, exist_ok=True)


def layer_stats(mean_abs, fire):
    m = np.sort(mean_abs)[::-1]
    return {"dead": float((fire < 0.001).mean()), "top1": float(m[:max(1, len(m) // 100)].sum() / m.sum()),
            "pr": float(m.sum() ** 2 / (len(m) * (m ** 2).sum())), "fire_median": float(np.median(fire))}


# ------------------------------------------------------------------------------------------- compare mode
if args.compare:
    runs = [dict(np.load(f, allow_pickle=True)) for f in args.compare]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    print(f"{'model':42}{'which':>6}{'layer':>6}{'N':>7}{'dead':>7}{'top1%':>7}{'PR':>6}{'fire':>7}")
    for r in runs:
        tag = str(r["tag"])
        for row, which in enumerate(("most", "least")):
            l = int(r["layers"][row])
            ma, fi = r["mean_abs"][l], r["fire"][l]
            s = layer_stats(ma, fi)
            print(f"{tag[:42]:42}{which:>6}{l:>6}{len(ma):>7}{s['dead']:7.3f}{s['top1']:7.3f}{s['pr']:6.2f}"
                  f"{s['fire_median']:7.3f}")
            x = np.linspace(0, 1, len(ma))
            axes[row, 0].plot(x, np.sort(ma)[::-1] / ma.mean(), label=f"{tag} (L{l})")
            axes[row, 1].plot(np.sort(fi), x, label=f"{tag} (L{l})")
        axes[0, 0].legend(fontsize=7)
    for row, which in enumerate(("most active layer", "least active layer")):
        axes[row, 0].set(yscale="log", xlabel="neuron rank / N", ylabel="mean |a| / layer mean", title=which)
        axes[row, 1].set(xscale="log", xlabel="firing rate", ylabel="share of neurons (CDF)", title=which)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"{ds}__compare.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    raise SystemExit

# ------------------------------------------------------------------------------------------- measure one model
model_id = args.model or cfg["teacher"]
chat = args.chat_template or (args.model is None and cfg.get("teacher_chat_template", False))
tag = os.path.basename(os.path.dirname(model_id.rstrip("/"))) if model_id.rstrip("/").endswith("/best") \
    else model_id.rstrip("/").split("/")[-1]
rows = load_split(cfg, cfg["datasets"][0], "validation")[:args.n]
tok = load_tokenizer(model_id)
model = load_model(model_id, "bf16", device_of(cfg))
texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
         if chat else r["prompt"] for r in rows]
tok.padding_side = tok.truncation_side = "left"

layers = model.model.layers
L, N = len(layers), layers[0].mlp.down_proj.in_features
acc = {k: torch.zeros(L, N, device=model.device) for k in ("abs", "sum", "fire")}
tau = torch.zeros(L, device=model.device)
per_prompt = np.zeros((len(texts), L, N), dtype=np.float16)   # mean |a| per prompt (for the density map)
state = {"mask": None, "i": 0, "tokens": 0}


def hook(l):
    def f(module, inp):
        a = inp[0].float()                                      # (batch, tokens, N)
        m = state["mask"]                                       # (batch, tokens, 1)
        if state["i"] == 0:
            tau[l] = ((a ** 2 * m).sum() / (m.sum() * N)).sqrt()
        acc["abs"][l] += (a.abs() * m).sum((0, 1))
        acc["sum"][l] += (a * m).sum((0, 1))
        acc["fire"][l] += ((a.abs() > tau[l]) * m).sum((0, 1))
        pp = (a.abs() * m).sum(1) / m.sum(1)
        per_prompt[state["i"]:state["i"] + len(a), l] = pp.cpu().numpy()
    return f


handles = [layer.mlp.down_proj.register_forward_pre_hook(hook(l)) for l, layer in enumerate(layers)]
ang = 0
with torch.no_grad():
    for i in range(0, len(texts), 8):
        enc = tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg["max_prompt_tokens"], add_special_tokens=not chat).to(model.device)
        m = enc["attention_mask"].clone()
        m[m.cumsum(1) == 1] = 0                                 # leave out the first real token (attention sink)
        state["mask"] = m[..., None].float()
        out = model(**enc, output_hidden_states=True)
        h = F.normalize(torch.stack(out.hidden_states)[:, :, -1].float(), dim=-1)
        ang = ang + (torch.arccos((h[:-1] * h[1:]).sum(-1).clamp(-1, 1)) / math.pi).sum(-1)
        state["i"] += len(enc["input_ids"])
        state["tokens"] += m.sum().item()
for hd in handles:
    hd.remove()

T = state["tokens"]
mean_abs = (acc["abs"] / T).cpu().numpy()
mean = (acc["sum"] / T).cpu().numpy()
fire = (acc["fire"] / T).cpu().numpy()
ang = (ang / len(texts)).cpu().numpy()
most, least = args.layers or (int(ang.argmax()), int(ang.argmin()))
out = os.path.join(args.out_dir, f"{ds}__{tag}")
np.savez_compressed(out + ".npz", tag=tag, dataset=ds, layers=[most, least], angular=ang, tau=tau.cpu().numpy(),
                    mean_abs=mean_abs, mean=mean, fire=fire, tokens=T,
                    map_most=per_prompt[:, most], map_least=per_prompt[:, least])

print(f"{tag}: {L} layers, {N} neurons/layer, {T} tokens; angular change per layer: {np.round(ang, 3).tolist()}")
fig, axes = plt.subplots(2, 3, figsize=(17, 9), gridspec_kw={"width_ratios": [2.2, 1, 1]})
fig.suptitle(f"{tag}  |  {ds}  |  {len(texts)} validation prompts, {T} tokens", fontsize=13)
for row, (which, l) in enumerate((("most", most), ("least", least))):
    s = layer_stats(mean_abs[l], fire[l])
    print(f"  {which:5} active layer {l:2}: angular {ang[l]:.3f}  dead {s['dead']:.3f}  top-1% share {s['top1']:.3f}  "
          f"PR {s['pr']:.2f}  median firing rate {s['fire_median']:.3f}")
    order = np.argsort(-mean_abs[l])
    pp = per_prompt[:, l][:, order].astype(np.float32)
    ax = axes[row, 0]
    im = ax.imshow(np.maximum(pp, 1e-4), aspect="auto", cmap="inferno", interpolation="nearest",
                   norm=LogNorm(vmin=max(pp[pp > 0].min(), 1e-4) if (pp > 0).any() else 1e-4, vmax=pp.max()))
    fig.colorbar(im, ax=ax, label="mean |a| over the prompt's tokens")
    ax.set(title=f"{which} active layer {l} (angular {ang[l]:.2f}): neurons sorted by mean |a|",
           xlabel="neuron (sorted)", ylabel="prompt")
    ax = axes[row, 1]
    x = np.arange(1, N + 1) / N
    ax.plot(x, mean_abs[l][order], color="tab:orange")
    ax.set(yscale="log", xlabel="neuron rank / N", ylabel="mean |a|",
           title=f"top 1%: {s['top1']:.0%} of activity, PR {s['pr']:.2f}")
    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(mean_abs[l][order]) / mean_abs[l].sum(), color="tab:blue", lw=1)
    ax2.set_ylabel("cumulative share", color="tab:blue")
    ax = axes[row, 2]
    ax.hist(np.clip(fire[l], 1e-4, 1), bins=np.logspace(-4, 0, 50), color="tab:purple")   # never firing: left edge
    ax.set(xscale="log", xlabel=f"firing rate (|a| > {tau[l].item():.3g})", ylabel="neurons",
           title=f"dead (< 0.1%): {s['dead']:.1%}, median {s['fire_median']:.3f}")
fig.tight_layout()
fig.savefig(out + ".png", dpi=150)
print(f"wrote {out}.png and {out}.npz")