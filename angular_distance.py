"""Which teacher layers to keep, by angular distance (Gromov et al., ICLR 2025). No training, no generation.

    python angular_distance.py --config configs/pubmedqa_llama_L7_chat.yaml

d(l, l+n) = arccos(cos(h_l, h_l+n)) / pi between the hidden state entering layer l and the one leaving layer
l+n-1, at the last prompt token, averaged over validation prompts. Small d = the block barely changes the state.
  block      Gromov: remove the one contiguous block of (32 - keep) layers with the smallest d
  iterative  remove the block of --block layers with the smallest d, re-measure the pruned teacher, repeat
The last layer is never removed (as in Gromov). Writes results/angular_distance.json.
"""
import csv
import json
import math
import os

import torch
import torch.nn.functional as F

from common import device_of, load_model, load_split, load_tokenizer, parse_args

args, cfg = parse_args(__doc__, lambda ap: (
    ap.add_argument("--n", type=int, default=300, help="number of validation prompts"),
    ap.add_argument("--keep", type=int, default=7),
    ap.add_argument("--block", type=int, default=5, help="layers removed per round (iterative)"),
    ap.add_argument("--out", default="results/angular_distance.json")))
rows = load_split(cfg, cfg["datasets"][0], "validation")[:args.n]
tok = load_tokenizer(cfg["teacher"])
model = load_model(cfg["teacher"], "bf16", device_of(cfg)).eval()
chat = cfg.get("teacher_chat_template", False)
texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
         if chat else r["prompt"] for r in rows]
layers = list(model.model.layers)


def use_layers(ids):
    keep = [layers[i] for i in ids]
    for k, layer in enumerate(keep):
        layer.self_attn.layer_idx = k
    model.model.layers = torch.nn.ModuleList(keep)
    model.config.num_hidden_layers = len(keep)


@torch.no_grad()
def distances():
    """D[i, j] = mean angular distance between hidden states i and j (0 = embeddings, i = after layer i)."""
    tok.padding_side = tok.truncation_side = "left"   # position -1 = last prompt token, even for long prompts
    D = 0
    for i in range(0, len(texts), 8):
        enc = tok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg["max_prompt_tokens"], add_special_tokens=not chat).to(model.device)
        h = F.normalize(torch.stack(model(**enc, output_hidden_states=True).hidden_states)[:, :, -1].float(), dim=-1)
        D = D + (torch.arccos(torch.einsum("ibd,jbd->ijb", h, h).clamp(-1, 1)) / math.pi).sum(-1).cpu()
    return D / len(texts)


def best_block(D, m, n):
    """Start of the n-layer block with the smallest d, among blocks that do not include the last layer."""
    return min(range(m - n), key=lambda l: D[l, l + n].item())


L = len(layers)
D = distances()
single = [round(D[l, l + 1].item(), 4) for l in range(L)]
n = L - args.keep
l = best_block(D, L, n)
block = [i for i in range(L) if not l <= i < l + n]

ids, trace = list(range(L)), []
while len(ids) > args.keep:
    use_layers(ids)
    Dp = distances() if len(ids) < L else D
    n = min(args.block, len(ids) - args.keep)
    l = best_block(Dp, len(ids), n)
    trace.append({"removed": ids[l:l + n], "distance": round(Dp[l, l + n].item(), 4)})
    print(f"remove layers {ids[l:l + n]} (d = {trace[-1]['distance']})")
    ids = ids[:l] + ids[l + n:]
use_layers(list(range(L)))

out = {"block": block, "iterative": ids, "iterative_trace": trace, "single_layer_distance": single,
       "distance_matrix": [[round(x, 4) for x in r] for r in D.tolist()]}
print(f"\nGromov one block : {block}\niterative        : {ids}")
if os.path.exists("results/layer_importance.csv"):   # compare with the data-based importance
    from scipy.stats import spearmanr
    drop = [float(r["acc_drop"]) for r in csv.DictReader(open("results/layer_importance.csv"))]
    data7 = sorted(sorted(range(L), key=lambda i: -drop[i])[:args.keep])
    rho, p = spearmanr(single, drop)
    out.update({"data_based": data7, "spearman_single_vs_acc_drop": [round(rho, 3), round(p, 3)]})
    print(f"data-based       : {data7}")
    print(f"overlap with data-based: block {len(set(block) & set(data7))}/{args.keep}, "
          f"iterative {len(set(ids) & set(data7))}/{args.keep}")
    print(f"Spearman d(l, l+1) vs accuracy drop: rho = {rho:.2f}, p = {p:.3f}")
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print(f"wrote {args.out}")