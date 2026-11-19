"""Importance of each teacher layer: remove one layer at a time (no training) and measure accuracy.

    python layer_importance.py --config configs/pubmedqa_llama_L7_chat.yaml --n 300
    python layer_importance.py --config configs/pubmedqa_llama_L7_chat.yaml --n 300 --split test --out results/layer_importance_test.csv

Default split is validation, so layers are never chosen on the test set.

Two measures per layer:
  acc_drop         accuracy of the full teacher minus accuracy with that layer removed (yes/no/maybe)
  block_influence  1 - cosine(input, output) of the layer, averaged over prompt tokens (ShortGPT);
                   cheap, no generation. Low = the layer barely changes the hidden state.
Writes results/layer_importance.csv.
"""
import csv
import os
import re

import torch
import torch.nn.functional as F

from common import device_of, generate, load_model, load_split, load_tokenizer, parse_args

args, cfg = parse_args(__doc__, lambda ap: (
    ap.add_argument("--n", type=int, default=300, help="number of questions"),
    ap.add_argument("--split", default="validation", help="validation (for choosing layers) or test (for reporting)"),
    ap.add_argument("--out", default="results/layer_importance.csv")))
rows = load_split(cfg, cfg["datasets"][0], args.split)[:args.n]
prompts = [r["prompt"] for r in rows]
tok = load_tokenizer(cfg["teacher"])
model = load_model(cfg["teacher"], "bf16", device_of(cfg)).eval()
chat = cfg.get("teacher_chat_template", False)
layers = list(model.model.layers)


def accuracy():
    outs = generate(model, tok, prompts, 8, 16, cfg["max_prompt_tokens"], chat=chat)   # label = first word
    labels = [re.search(r"\b(yes|no|maybe)\b", o["text"].lower()) for o in outs]
    return sum(bool(m) and m.group(1) == r["gold"] for m, r in zip(labels, rows)) / len(rows)


def use_layers(keep):
    for k, layer in enumerate(keep):
        layer.self_attn.layer_idx = k          # keep the KV cache consistent
    model.model.layers = torch.nn.ModuleList(keep)
    model.config.num_hidden_layers = len(keep)


@torch.no_grad()
def block_influence(batch=8):
    bi = torch.zeros(len(layers))
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg["max_prompt_tokens"]).to(model.device)
        h = model(**enc, output_hidden_states=True).hidden_states   # h[j] = input of layer j
        m = enc["attention_mask"].bool()
        for j in range(len(layers)):
            bi[j] += (1 - F.cosine_similarity(h[j][m].float(), h[j + 1][m].float(), dim=-1)).mean().cpu()
    return bi / -(-len(prompts) // batch)


base = accuracy()
bi = block_influence()
print(f"full model ({len(layers)} layers): acc {base:.3f}")
results = []
for i in range(len(layers)):
    use_layers(layers[:i] + layers[i + 1:])
    acc = accuracy()
    results.append({"layer": i, "acc": round(acc, 4), "acc_drop": round(base - acc, 4),
                    "block_influence": round(float(bi[i]), 4)})
    print(results[-1])
use_layers(layers)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0])
    w.writeheader()
    w.writerows(results)
print(f"wrote {args.out}; most important layers (largest acc drop):",
      [r["layer"] for r in sorted(results, key=lambda r: -r["acc_drop"])[:7]])