"""Quick pass@1 check on a few code test problems, on the CPU (does not touch the GPU used for training).

    python quick_code_eval.py --model runs/code_qwen25coder-7b-p7_kd/best --n 20
    python quick_code_eval.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --chat-template --n 10

Problems are spread evenly over the test split (HumanEval+ first, then MBPP+). Slow: about 1-3 min per problem.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only, set before torch is imported

import argparse
import collections
import json

from common import generate, load_model, load_tokenizer
from evaluate import task_scores

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/code_qwen25coder-7b-p7_kd/best")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--chat-template", action="store_true", help="for original HF models; not for trained students")
args = ap.parse_args()

rows = [json.loads(l) for l in open("data/splits/code/test.jsonl")]
rows = [rows[i * len(rows) // args.n] for i in range(args.n)]
tok, model = load_tokenizer(args.model), load_model(args.model, "fp32", "cpu")

results = []
for r in rows:
    out = generate(model, tok, [r["prompt"]], 512, batch_size=1, chat=args.chat_template)[0]
    res = task_scores(r, out["text"])["result"]
    results.append(res)
    print(f"{r['id']:28} {res:14} ({out['n_new_tokens']} tokens)", flush=True)

print(f"\npass@1 = {results.count('pass')}/{len(results)} = {results.count('pass') / len(results):.2f}")
print(dict(collections.Counter(results)))