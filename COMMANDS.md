# D3LM commands

Run everything from `~/Development/d3lm-ccgrid`. All runs use the same config:

```bash
CFG=configs/pubmedqa_llama_L7_chat.yaml
```

Use `tmux` (or `nohup ... > logs/<name>.log 2>&1 &`) for long runs.

## 1. Data: splits + teacher answers

```bash
python prepare_data.py --config $CFG --vllm
```

Check that the teacher answers finished (`stopped` should be close to the total):

```bash
python -c "
import json
for s in ('train','validation'):
    r=[json.loads(l) for l in open(f'data/teacher/llama31-8b-it-chat/pubmedqa_{s}.jsonl')]
    print(s, len(r), 'stopped:', sum(x['stopped'] for x in r))"
```

If almost none are stopped, the files are old: `rm -rf data/teacher/llama31-8b-it-chat`, then regenerate.

## 2. Teacher reference (run before any student evaluation)

```bash
python evaluate.py --config $CFG --name teacher_chat --model meta-llama/Llama-3.1-8B-Instruct --chat-template
```

## 3. Main method: KD (forward KL + CE)

```bash
python train.py --config $CFG --set run_name=chat_L7
python evaluate.py --config $CFG --name chat_L7 --model runs/chat_L7/best --reference teacher_chat
```

## 4. Baselines with training

```bash
# SFT on teacher answers (sequence-level KD)
python train.py --config $CFG --set run_name=sft_teacher_L7 train.objective=sft train.targets=teacher
python evaluate.py --config $CFG --name sft_teacher_L7 --model runs/sft_teacher_L7/best --reference teacher_chat

# Reverse KL (MiniLLM loss)
python train.py --config $CFG --set run_name=rkl_L7 train.divergence=rkl
python evaluate.py --config $CFG --name rkl_L7 --model runs/rkl_L7/best --reference teacher_chat

# Skew KL (DistiLLM loss)
python train.py --config $CFG --set run_name=skl_L7 train.divergence=skl
python evaluate.py --config $CFG --name skl_L7 --model runs/skl_L7/best --reference teacher_chat

# Optional: SFT on gold answers ("why use a teacher at all?")
python train.py --config $CFG --set run_name=sft_gold_L7 train.objective=sft train.targets=gold
python evaluate.py --config $CFG --name sft_gold_L7 --model runs/sft_gold_L7/best --reference teacher_chat
```

## 5. Baselines without training

```bash
# Quantized teacher
python evaluate.py --config $CFG --name teacher_int8 --model meta-llama/Llama-3.1-8B-Instruct --precision int8 --chat-template --reference teacher_chat
python evaluate.py --config $CFG --name teacher_nf4  --model meta-llama/Llama-3.1-8B-Instruct --precision nf4  --chat-template --reference teacher_chat

# Pruned 7-layer teacher, untrained (1 step with lr=0 saves the initial model)
python train.py --config $CFG --set run_name=pruned_L7_untrained train.max_steps=1 train.lr=0 train.eval_every=1
python evaluate.py --config $CFG --name pruned_L7_untrained --model runs/pruned_L7_untrained/best --reference teacher_chat

# Published chat models
for m in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
  python evaluate.py --config $CFG --name $(basename $m) --model $m --chat-template --reference teacher_chat
done

# Published base models (not chat-tuned: no --chat-template)
# Sheared-LLaMA uses another tokenizer: its ppl is not comparable, use acc + text metrics only
for m in nvidia/Llama-3.1-Minitron-4B-Width-Base princeton-nlp/Sheared-LLaMA-1.3B princeton-nlp/Sheared-LLaMA-2.7B; do
  python evaluate.py --config $CFG --name $(basename $m) --model $m --reference teacher_chat
done
```

## 6. Seeds (for mean ± std; repeat for each trained run)

```bash
for s in 2 3; do
  python train.py --config $CFG --set run_name=chat_L7_s$s seed=$s
  python evaluate.py --config $CFG --name chat_L7_s$s --model runs/chat_L7_s$s/best --reference teacher_chat
  python train.py --config $CFG --set run_name=sft_teacher_L7_s$s seed=$s train.objective=sft train.targets=teacher
  python evaluate.py --config $CFG --name sft_teacher_L7_s$s --model runs/sft_teacher_L7_s$s/best --reference teacher_chat
done
```

## 6b. CE-weight sweep (loss = (1 - w) * KD + w * CE)

```bash
# skips weights already evaluated; 1.0 = SFT on teacher answers (sft_teacher_L7, section 4)
for w in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  [ -f eval/ce${w}_L7/pubmedqa/metrics.json ] && { echo "skip $w"; continue; }
  python train.py --config $CFG --set run_name=ce${w}_L7 train.ce_weight=$w
  python evaluate.py --config $CFG --name ce${w}_L7 --model runs/ce${w}_L7/best --reference teacher_chat
done

python - <<'EOF'
import json, csv, os
os.makedirs("results", exist_ok=True)
runs = [(f"{w / 10:.1f}", f"ce{w / 10:.1f}_L7") for w in range(10)] + [("1.0", "sft_teacher_L7")]
rows = []
for w, run in runs:
    if not os.path.exists(f"eval/{run}/pubmedqa/metrics.json"):
        print("missing", run); continue
    m = json.load(open(f"eval/{run}/pubmedqa/metrics.json"))
    s = json.load(open(f"runs/{run}/summary.json"))
    rows.append({"ce_weight": w, "best_val_ce": round(s["best_val_ce"], 4), "acc": round(m["acc"]["mean"], 4),
                 "macro_f1": round(m["macro_f1"], 4), "fid_rougeL": round(m["fid_rougeL"]["mean"], 4),
                 "fid_ppl": round(m["fid_ppl"]["mean"], 3)})
with open("results/ce_weight.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
print("best by acc:", max(rows, key=lambda r: r["acc"])["ce_weight"])
EOF
```

The table is saved in `results/ce_weight.csv`. View it later with `column -s, -t results/ce_weight.csv`.
Note: `best_val_ce` of the SFT run (1.0) is plain CE, comparable to the others. Report the trend as a sensitivity analysis (it is measured on the test set).

## 7. Compare results

```bash
python - teacher_chat chat_L7 sft_teacher_L7 <<'EOF'
import json, sys
runs = sys.argv[1:]
def flat(m):
    out = {}
    for k, v in m.items():
        if isinstance(v, dict) and "mean" in v: out[k] = v["mean"]
        elif isinstance(v, dict): out.update({f"{k}.{a}": b for a, b in v.items() if isinstance(b, (int, float))})
        elif isinstance(v, (int, float)): out[k] = v
    return out
ms = [flat(json.load(open(f"eval/{r}/pubmedqa/metrics.json"))) for r in runs]
print(f"{'metric':32}" + "".join(f"{r:>16}" for r in runs))
for k in sorted(set().union(*ms)):
    print(f"{k:32}" + "".join(f"{m[k]:16.4f}" if k in m else f"{'-':>16}" for m in ms))
EOF
```

Paired significance test on accuracy (McNemar; change the two run names):

```bash
python - chat_L7 sft_teacher_L7 <<'EOF'
import json, sys
from scipy.stats import binomtest
A, B = sys.argv[1:3]
load = lambda r: {json.loads(l)["id"]: json.loads(l)["acc"] for l in open(f"eval/{r}/pubmedqa/predictions.jsonl")}
a, b = load(A), load(B)
x = sum(a[i] and not b[i] for i in a); y = sum(b[i] and not a[i] for i in a)
print(f"{A} right only: {x}  {B} right only: {y}  p = {binomtest(x, x + y).pvalue:.2g}")
EOF
```

## 7b. Qualitative examples (10 random test cases, from saved predictions)

```bash
python qualitative.py ce0.3_L7 teacher_chat sft_teacher_L7 --n 10 --out results/qualitative.md
```

## 8. Efficiency: energy, speed, memory (nothing else on the GPU)

Each run measures idle power, then generates 128 tokens for 256 test prompts at batch sizes 1, 8, 32
(3 repeats, >=30 s each). Output: `bench/<name>.json`. About 10-20 min per model.

```bash
python energy_benchmark.py --config $CFG --name teacher     --model meta-llama/Llama-3.1-8B-Instruct
python energy_benchmark.py --config $CFG --name teacher_nf4 --model meta-llama/Llama-3.1-8B-Instruct --precision nf4
python energy_benchmark.py --config $CFG --name student     --model runs/ce0.3_L7/best
python energy_benchmark.py --config $CFG --name llama32_1b  --model meta-llama/Llama-3.2-1B-Instruct
```

Summary table (student vs teacher, per batch size):

```bash
python - teacher teacher_nf4 student llama32_1b <<'EOF'
import json, sys
names = sys.argv[1:]
b = {n: {r["batch_size"]: r["mean"] for r in json.load(open(f"bench/{n}.json"))["hf"]} for n in names}
for bs in sorted(b[names[0]]):
    t = b[names[0]][bs]
    print(f"\nbatch {bs}")
    print(f"{'model':14}{'tok/s':>9}{'J/req':>9}{'J/tok':>8}{'mem GB':>8}{'speedup':>9}{'energy x':>10}")
    for n in names:
        m = b[n][bs]
        print(f"{n:14}{m['tokens_per_s']:9.0f}{m['j_per_request']:9.1f}{m['j_per_token']:8.3f}{m['peak_mem_gb']:8.1f}"
              f"{m['tokens_per_s'] / t['tokens_per_s']:9.2f}{t['j_per_request'] / m['j_per_request']:10.2f}")
EOF
```

`speedup` = tok/s relative to the first model (teacher); `energy x` = how many times less energy per request.

Parameter count of the student:

```bash
python -c "
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('runs/ce0.3_L7/best')
print(f'{sum(p.numel() for p in m.parameters()) / 1e9:.2f}B parameters')"
```

Lifecycle energy and break-even (data generation + ce-weight search + training vs. per-request savings):

```bash
python break_even.py --spec configs/break_even.yaml
```

Output: `results/break_even.csv` and a cumulative-energy plot in `results/`.

## 8b. GPU compute measurements

**Serving under load (vLLM):** latency (TTFT, TPOT, p50/p99), throughput and max request rate under the SLO
(TTFT <= 1 s, TPOT <= 0.1 s). Output: `bench/<name>_vllm.json`.

```bash
python energy_benchmark.py --config $CFG --name teacher_vllm --model meta-llama/Llama-3.1-8B-Instruct --vllm
python energy_benchmark.py --config $CFG --name student_vllm --model runs/ce0.3_L7/best --vllm

python - teacher_vllm student_vllm <<'EOF'
import json, sys
for n in sys.argv[1:]:
    b = json.load(open(f"bench/{n}.json"))
    print(f"\n{n}: offline {b['offline']['tokens_per_s']:.0f} tok/s, max rate under SLO: {b['max_rate_under_slo']} req/s")
    print(f"{'rate':>6}{'tok/s':>9}{'TTFT p50':>10}{'TTFT p99':>10}{'TPOT p50':>10}{'e2e p99':>9}{'SLO ok':>8}{'J/req':>8}")
    for r in b["online"]:
        print(f"{r['offered_rps']:6.0f}{r['tokens_per_s']:9.0f}{r['ttft_p50']:10.3f}{r['ttft_p99']:10.3f}"
              f"{r['tpot_p50']:10.4f}{r['e2e_p99']:9.2f}{r['slo_ok']:8.0%}{r.get('j_per_request', 0):8.1f}")
EOF
```

**GPU utilisation, power, memory while benchmarking** (run in a second terminal; stop with Ctrl-C):

```bash
nvidia-smi dmon -s pucm -d 1 -o T > logs/dmon_student.txt     # sm %, mem %, power W, clocks, every second
```

**Static compute cost per model** (no GPU needed): parameters, weight memory, FLOPs per token, KV-cache per token.

```bash
python - meta-llama/Llama-3.1-8B-Instruct runs/ce0.3_L7/best <<'EOF'
import sys
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
for name in sys.argv[1:]:
    c = AutoConfig.from_pretrained(name)
    with init_empty_weights():          # builds the model without loading weights
        m = AutoModelForCausalLM.from_config(c)
    n = sum(p.numel() for p in m.parameters())
    kv = 2 * c.num_hidden_layers * c.num_key_value_heads * (c.hidden_size // c.num_attention_heads) * 2   # K+V, bf16
    print(f"{name}: {c.num_hidden_layers} layers, {n / 1e9:.2f}B params, {n * 2 / 2**30:.1f} GB weights (bf16), "
          f"{2 * n / 1e9:.1f} GFLOPs/token, KV cache {kv / 1024:.0f} KB/token")
EOF
```

FLOPs/token = 2 x params (forward pass, excluding attention over the context). A smaller KV cache per token
means more concurrent requests fit in GPU memory, which is why the student's max serving rate is higher.

## 9. Push code to GitHub

Once, in `d3lm-ccgrid/` (code + configs only; data, models and the venv are ignored):

```bash
cat > .gitignore <<'EOF'
data/
runs/
bench/
logs/
.venv/
__pycache__/
*.pyc
EOF
git init
git add .
git commit -m "D3LM CCGrid code"
git branch -M main
git remote add origin https://github.com/QCDIS/D3LM.git
git push -u origin main
```

If the push is rejected because the new repo already has a README/license, overwrite it (you are the only user):
`git push -u origin main --force`

After that, every update:

```bash
git add .
git commit -m "<what changed>"
git push
```

`eval/` and `results/` are small and are committed (they hold the paper's numbers). Remove them from git with
`echo -e "eval/\nresults/" >> .gitignore` if you prefer. Keep the repo private until the double-blind review ends.

## 10. Disk

```bash
du -sh runs/*/best | sort -h
df -h /
rm -rf runs/<run>/best        # keeps summary/logs; eval/ results are not affected
```

## 11. Ask the model in the terminal

```bash
python chat.py                                    # best student (runs/ce0.3_L7/best)
python chat.py --model meta-llama/Llama-3.1-8B-Instruct --chat-template   # teacher, for comparison
```

## 12. Student screening

Same KD + CE training as the main student. Small models: all layers, fp32. 8B models: pruned to 7 layers,
current settings. Non-Llama models use another tokenizer, so they train on CE only (same teacher answers).

```bash
bash screen_students.sh 2>&1 | tee logs/screen_students.log
```

Summary (all students, ranked by accuracy):

```bash
python - <<'EOF'
import json, glob, os
rows = []
for f in glob.glob("eval/*/pubmedqa/metrics.json"):
    run = f.split("/")[1]
    if not (run.startswith("pubmedqa_") or run in ("ce0.3_L7", "teacher_chat")):
        continue
    m = json.load(open(f))
    s = f"runs/{run}/summary.json"
    params = json.load(open(s))["student"]["params_total"] / 1e9 if os.path.exists(s) else float("nan")
    rows.append((run, params, m["acc"]["mean"], m["macro_f1"], m["fid_rougeL"]["mean"], m["fid_ppl"]["mean"]))
print(f"{'run':40}{'params B':>9}{'acc':>7}{'F1':>7}{'fid_rL':>8}{'fid_ppl':>9}")
for r in sorted(rows, key=lambda r: -r[2]):
    print(f"{r[0]:40}{r[1]:9.2f}{r[2]:7.3f}{r[3]:7.3f}{r[4]:8.3f}{r[5]:9.2f}")
EOF
```

`fid_ppl` is only comparable between models with the same tokenizer. Pick the best student by acc / macro-F1
per parameter, then benchmark its energy (section 8) and use it for the other training types.