# D3LM commands

All experiments use 1500 training steps (decided 18 Nov 2026); earlier `_3000st` runs are kept for reference only.

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

## 7c. Method experiments table (one row per experiment, order of EXPERIMENTS.md)

```bash
python - <<'EOF'
import json, os
runs = [("1  SFT only", "pubmedqa_llama31-8b-p7_sft_1500st"),
        ("3  KD+CE fkl (= 5c)", "pubmedqa_llama31-8b-p7_kd_1500st"),
        ("4a KD+CE rkl", "pubmedqa_llama31-8b-p7_kdrkl_1500st"),
        ("4b KD+CE skl", "pubmedqa_llama31-8b-p7_kdskl_1500st"),
        ("5b gradual KD+CE", "pubmedqa_llama31-8b-p14-p7_kd_1500st"),
        ("6  one-step KD+CE+hidden", "pubmedqa_llama31-8b-p7_kdhid_1500st"),
        ("6b one-step hidden only", "pubmedqa_llama31-8b-p7_hid_1500st"),
        ("7  gradual KD+CE+hidden", "pubmedqa_llama31-8b-p14-p7_kdhid_1500st"),
        ("   teacher", "teacher_chat")]
print(f"{'experiment':30}{'acc':>7}{'F1':>7}{'fid_rL':>8}{'fid_ppl':>9}{'best/steps':>12}")
for label, r in runs:
    f = f"eval/{r}/pubmedqa/metrics.json"
    if not os.path.exists(f):
        print(f"{label:30}{'not run':>7}"); continue
    m = json.load(open(f))
    g = lambda k: m[k]["mean"] if isinstance(m.get(k), dict) else m.get(k, float("nan"))
    s = f"runs/{r}/summary.json"
    st = (lambda d: f"{d['best_step']}/{d['steps']}")(json.load(open(s))) if os.path.exists(s) else "-"
    print(f"{label:30}{g('acc'):7.3f}{g('macro_f1'):7.3f}{g('fid_rougeL'):8.3f}{g('fid_ppl'):9.2f}{st:>12}".replace("nan", "  -"))
EOF
```

## 7d. All evaluated runs, sorted by accuracy

```bash
python - <<'EOF'
import json, glob, os
rows = []
for f in glob.glob("eval/*/pubmedqa/metrics.json"):
    run = f.split("/")[1]
    m = json.load(open(f))
    g = lambda k: m[k]["mean"] if isinstance(m.get(k), dict) else m.get(k, float("nan"))
    s = f"runs/{run}/summary.json"
    st = (lambda d: f"{d['best_step']}/{d['steps']}")(json.load(open(s))) if os.path.exists(s) else "-"
    rows.append((run, g("acc"), g("macro_f1"), g("fid_rougeL"), g("fid_ppl"), st))
print(f"{'run':48}{'acc':>7}{'F1':>7}{'fid_rL':>8}{'fid_ppl':>9}{'best/steps':>12}")
for r in sorted(rows, key=lambda r: -r[1]):
    print(f"{r[0]:48}{r[1]:7.3f}{r[2]:7.3f}{r[3]:8.3f}{r[4]:9.2f}{r[5]:>12}".replace("nan", "  -"))
EOF
```

## 7e. Every metric, for chosen runs (and all runs saved to results/all_metrics.csv)

```bash
python - pubmedqa_llama31-8b-p14-p7_kdhid_1500st pubmedqa_llama32-1b_kd pubmedqa_qwen25-1.5b_ce teacher_chat <<'EOF'
import json, glob, os, sys, csv
def flat(m, pre=""):
    out = {}
    for k, v in m.items():
        if isinstance(v, dict) and "mean" in v: out[pre + k] = v["mean"]
        elif isinstance(v, dict): out.update(flat(v, pre + k + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool): out[pre + k] = v
    return out
allm = {}
for f in sorted(glob.glob("eval/*/pubmedqa/metrics.json")):
    run = f.split("/")[1]
    allm[run] = flat(json.load(open(f)))
    s = f"runs/{run}/summary.json"
    if os.path.exists(s):
        d = json.load(open(s))
        allm[run].update({"train.best_step": d["best_step"], "train.steps": d["steps"],
                          "train.best_val_ce": d["best_val_ce"], "train.kwh": d.get("energy", {}).get("kwh", float("nan"))})
keys = sorted(set().union(*allm.values()))
os.makedirs("results", exist_ok=True)
with open("results/all_metrics.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["run"] + keys)
    for run, m in allm.items(): w.writerow([run] + [m.get(k, "") for k in keys])
runs = sys.argv[1:] or list(allm)
print(f"{'metric':38}" + "".join(f"{r[:22]:>24}" for r in runs))
for k in keys:
    print(f"{k:38}" + "".join(f"{allm.get(r, {}).get(k, float('nan')):24.4f}" for r in runs).replace("nan", "  -"))
print("\nall runs and metrics saved to results/all_metrics.csv")
EOF
```

## 7f. Every metric for one dataset (climate, climate_long, code, pubmedqa)

```bash
python - climate <<'EOF'
import json, glob, os, sys, csv
DS = sys.argv[1]
def flat(m, pre=""):
    out = {}
    for k, v in m.items():
        if isinstance(v, dict) and "mean" in v: out[pre + k] = v["mean"]
        elif isinstance(v, dict): out.update(flat(v, pre + k + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool): out[pre + k] = v
    return out
allm = {}
for f in sorted(glob.glob(f"eval/*/{DS}/metrics.json")):
    run = f.split("/")[1]
    allm[run] = flat(json.load(open(f)))
    s = f"runs/{run}/summary.json"
    if os.path.exists(s):
        d = json.load(open(s))
        allm[run].update({"train.best_step": d["best_step"], "train.steps": d["steps"],
                          "train.best_val_ce": d["best_val_ce"], "train.kwh": d.get("energy", {}).get("kwh", float("nan"))})
keys = sorted(set().union(*allm.values()))
runs = list(allm)
print(f"{'metric':36}" + "".join(f"{r[:24]:>26}" for r in runs))
for k in keys:
    print(f"{k:36}" + "".join(f"{allm[r].get(k, float('nan')):26.4f}" for r in runs).replace("nan", "  -"))
os.makedirs("results", exist_ok=True)
with open(f"results/all_metrics_{DS}.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["run"] + keys)
    for r in runs: w.writerow([r] + [allm[r].get(k, "") for k in keys])
print(f"\nsaved to results/all_metrics_{DS}.csv")
EOF
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
    g = lambda k: m[k]["mean"] if isinstance(m.get(k), dict) else m.get(k, float("nan"))
    s = f"runs/{run}/summary.json"
    params = json.load(open(s))["student"]["params_total"] / 1e9 if os.path.exists(s) else float("nan")
    rows.append((run, params, g("acc"), g("macro_f1"), g("fid_rougeL"), g("fid_ppl")))
print(f"{'run':40}{'params B':>9}{'acc':>7}{'F1':>7}{'fid_rL':>8}{'fid_ppl':>9}")
for r in sorted(rows, key=lambda r: -r[2]):
    print(f"{r[0]:40}{r[1]:9.2f}{r[2]:7.3f}{r[3]:7.3f}{r[4]:8.3f}{r[5]:9.2f}".replace("nan", "  -"))
EOF
```

`fid_ppl` is only comparable between models with the same tokenizer. Pick the best student by acc / macro-F1
per parameter, then benchmark its energy (section 8) and use it for the other training types.

## 13. Gradual pruning (32 -> 14 -> 7) and hidden-state distillation

```bash
# stage 1: 32 -> 14 layers, CE only (teacher not loaded, so it fits in memory)
python train.py --config $CFG --set run_name=pubmedqa_llama31-8b-p14_ce_1500st student.num_layers=14 \
    train.objective=sft train.optimizer=adamw_8bit train.batch_size=2 train.grad_accum=16 train.max_steps=1500

# stage 2: 14 -> 7 layers, KD + CE
python train.py --config $CFG --set run_name=pubmedqa_llama31-8b-p14-p7_kd_1500st \
    student.source=runs/pubmedqa_llama31-8b-p14_ce_1500st/best student.num_layers=7 train.ce_weight=0.3 train.max_steps=1500
python evaluate.py --config $CFG --name pubmedqa_llama31-8b-p14-p7_kd_1500st \
    --model runs/pubmedqa_llama31-8b-p14-p7_kd_1500st/best --reference teacher_chat

# stage 2 with hidden-state distillation (each student layer matches the output of a group of teacher layers)
python train.py --config $CFG --set run_name=pubmedqa_llama31-8b-p14-p7_kdhid_1500st \
    student.source=runs/pubmedqa_llama31-8b-p14_ce_1500st/best student.num_layers=7 train.ce_weight=0.3 \
    train.max_steps=1500 train.hidden_weight=1.0
python evaluate.py --config $CFG --name pubmedqa_llama31-8b-p14-p7_kdhid_1500st \
    --model runs/pubmedqa_llama31-8b-p14-p7_kdhid_1500st/best --reference teacher_chat
```

## 14. One-step 32 -> 7: hidden-state loss only, and the matching KD + CE baseline

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
# hidden-state loss only (no KD, no CE)
python train.py --config $CFG --set run_name=pubmedqa_llama31-8b-p7_hid_1500st student.num_layers=7 \
    train.objective=hidden train.max_steps=1500 train.batch_size=2 train.grad_accum=16
python evaluate.py --config $CFG --name pubmedqa_llama31-8b-p7_hid_1500st \
    --model runs/pubmedqa_llama31-8b-p7_hid_1500st/best --reference teacher_chat

# 5c: KD + CE, same steps
python train.py --config $CFG --set run_name=pubmedqa_llama31-8b-p7_kd_1500st student.num_layers=7 \
    train.ce_weight=0.3 train.max_steps=1500 train.batch_size=2 train.grad_accum=16
python evaluate.py --config $CFG --name pubmedqa_llama31-8b-p7_kd_1500st \
    --model runs/pubmedqa_llama31-8b-p7_kd_1500st/best --reference teacher_chat
```

## 15. All methods at 1500 steps (same settings for every run)

7 layers, ce_weight 0.3, 1500 steps, batch 2 x accum 16. Runs that already have results are skipped.

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
CFG=configs/pubmedqa_llama_L7_chat.yaml
S="student.num_layers=7 train.ce_weight=0.3 train.max_steps=1500 train.batch_size=2 train.grad_accum=16"
P14=runs/pubmedqa_llama31-8b-p14_ce_1500st/best        # stage 1 of gradual pruning (32 -> 14), already trained

te() {   # train + evaluate, skipped if already evaluated
  [ -f eval/$1/pubmedqa/metrics.json ] && { echo "skip $1"; return; }
  python train.py --config $CFG --set run_name=$1 $S "${@:2}" 2>&1 | tee logs/train_$1.log
  python evaluate.py --config $CFG --name $1 --model runs/$1/best --reference teacher_chat 2>&1 | tee logs/eval_$1.log
}
mkdir -p logs
te pubmedqa_llama31-8b-p7_sft_1500st      train.objective=sft                          # 1  SFT only
te pubmedqa_llama31-8b-p7_kd_1500st                                                    # 3  KD+CE fkl (= 5c)
te pubmedqa_llama31-8b-p7_kdrkl_1500st    train.divergence=rkl                         # 4a KD+CE rkl
te pubmedqa_llama31-8b-p7_kdskl_1500st    train.divergence=skl                         # 4b KD+CE skew KL
te pubmedqa_llama31-8b-p14-p7_kd_1500st   student.source=$P14                          # 5  gradual KD+CE
te pubmedqa_llama31-8b-p7_kdhid_1500st    train.hidden_weight=1.0                      # 6  KD+CE+hidden
te pubmedqa_llama31-8b-p7_hid_1500st      train.objective=hidden                       # 6b hidden only
te pubmedqa_llama31-8b-p14-p7_kdhid_1500st student.source=$P14 train.hidden_weight=1.0 # 7  gradual KD+CE+hidden
```

## 16. Layer importance of the teacher, and which 7 layers the student keeps

Removes one teacher layer at a time and measures accuracy on 300 validation questions (not the test set,
so the layer choice is not tuned on test), plus the ShortGPT block influence. About 30-60 min.
Output: `results/layer_importance.csv`. Background: project doc `claude/layer_importance_related_work.md`.

```bash
python layer_importance.py --config configs/pubmedqa_llama_L7_chat.yaml --n 300
```

Result (300 validation questions, full teacher 0.910): layers 0 and 1 are essential (accuracy 0.0 without them),
then 30 (drop 0.113), 31, 11, 15 (about 0.06) and 9 (0.037); the other 25 layers are within noise (about ±0.03).

Layer-selection ablation, same settings as the evenly spaced one-step run (`pubmedqa_llama31-8b-p7_kd_1500st`, 0.633):

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
S="student.num_layers=7 train.ce_weight=0.3 train.max_steps=1500 train.batch_size=2 train.grad_accum=16"
sel() {   # sel <run name> <layer ids>: train + evaluate
  python train.py --config $CFG --set run_name=$1 $S "student.layer_ids=$2" "${@:3}"
  python evaluate.py --config $CFG --name $1 --model runs/$1/best --reference teacher_chat
}
sel pubmedqa_llama31-8b-p7imp_kd_1500st  "[0,1,9,11,15,30,31]"   # 8a: the 7 layers with the largest measured drop
sel pubmedqa_llama31-8b-p7edge_kd_1500st "[0,1,2,3,29,30,31]"    # 8b (optional): first 4 + last 3
sel pubmedqa_llama31-8b-p7bot_kd_1500st  "[0,1,2,3,4,5,31]"      # 8c (optional): bottom 6 + last
sel pubmedqa_llama31-8b-p7imp_kdhid_1500st "[0,1,9,11,15,30,31]" train.hidden_weight=1.0   # 9: importance layers + hidden loss
```

## 17. Data-free layer statistics vs measured importance

Run section 16 first (it writes `results/layer_importance.csv`). Then, about 2-5 min on the GPU:

```bash
python weight_stats.py meta-llama/Llama-3.1-8B-Instruct --importance results/layer_importance.csv
```

Prints out_norm, stable_rank and alpha per layer, then the Spearman correlation of each (and of block
influence) with the accuracy drop. Output: `results/weight_stats.csv`.

## 18. Gradual pruning with importance-selected layers (32 -> 14 -> 7)

Stage 1 keeps the 14 teacher layers with the largest accuracy drop (section 16):
[0,1,5,7,9,11,14,15,21,22,23,29,30,31]. Stage 2 keeps the 7 most important of them. Layer ids in stage 2
count positions in the 14-layer model: teacher layers [0,1,9,11,15,30,31] are positions [0,1,4,5,7,12,13].

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
P14I=runs/pubmedqa_llama31-8b-p14imp_ce_1500st/best

# stage 1: 32 -> 14 importance layers, CE only
python train.py --config $CFG --set run_name=pubmedqa_llama31-8b-p14imp_ce_1500st student.num_layers=14 \
    "student.layer_ids=[0,1,5,7,9,11,14,15,21,22,23,29,30,31]" \
    train.objective=sft train.optimizer=adamw_8bit train.batch_size=2 train.grad_accum=16 train.max_steps=1500

# stage 2: 14 -> 7, KD + CE (option A: the 7 important teacher layers, no new measurement)
python train.py --config $CFG --set run_name=pubmedqa_llama31-8b-p14imp-p7imp_kd_1500st student.source=$P14I \
    student.num_layers=7 "student.layer_ids=[0,1,4,5,7,12,13]" \
    train.ce_weight=0.3 train.max_steps=1500 train.batch_size=2 train.grad_accum=16
python evaluate.py --config $CFG --name pubmedqa_llama31-8b-p14imp-p7imp_kd_1500st \
    --model runs/pubmedqa_llama31-8b-p14imp-p7imp_kd_1500st/best --reference teacher_chat

# option B (optional): re-measure importance on the 14-layer model, then pick its 7 most important positions
# (the student was trained without the chat template, so it is measured without it)
python layer_importance.py --config $CFG --n 300 --set teacher=$P14I teacher_chat_template=false \
    --out results/layer_importance_p14imp.csv
```

## 19. Angular distance (Gromov et al.) vs measured importance, no training

Run section 16 first. About 1 min for the measurement plus 4 re-measurements for the iterative selection.

```bash
python angular_distance.py --config configs/pubmedqa_llama_L7_chat.yaml
```

Prints the 7 layers kept by Gromov's one-block removal and by iterative block removal (5 layers per round, re-measured),
their overlap with the data-based 7, and the Spearman correlation of single-layer distance with the accuracy drop.
Output: `results/angular_distance.json`. Train the selections with the `sel` function from section 16:

```bash
sel pubmedqa_llama31-8b-p7angblk_kd_1500st "[...block...]"       # 8d: Gromov one block
sel pubmedqa_llama31-8b-p7angit_kd_1500st  "[...iterative...]"   # 8e: iterative angular distance
```

## 20. Code domain (Python): Qwen2.5-Coder-7B-Instruct teacher, HumanEval+ / MBPP+

Train/validation prompts: `bigcode/self-oss-instruct-sc2-exec-filter-50k` (20k / 500); test: HumanEval+ (164) +
MBPP+ (378), scored by running their tests (pass@1, greedy). Model-written code is executed: run on the VM only.

```bash
CFG=configs/code_qwen_L7.yaml

# 1. splits + teacher answers (chat template, up to 512 tokens), then check how many were truncated
python prepare_data.py --config $CFG --vllm
python -c "
import json
for s in ('train','validation'):
    r=[json.loads(l) for l in open(f'data/teacher/qwen25-coder-7b-it-chat/code_{s}.jsonl')]
    print(s, len(r), 'stopped:', sum(x['stopped'] for x in r))"

# 2. teacher and zero-shot small model (both with the chat template)
python evaluate.py --config $CFG --name code_teacher --model Qwen/Qwen2.5-Coder-7B-Instruct --chat-template --metrics chrf
python evaluate.py --config $CFG --name code_qwen25coder-1.5b_zs --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --chat-template --reference code_teacher --metrics chrf

# 3. students (KD + CE): pruned 7-layer teacher, and the 1.5B coder model (same tokenizer)
python train.py --config $CFG --set run_name=code_qwen25coder-7b-p7_kd
python evaluate.py --config $CFG --name code_qwen25coder-7b-p7_kd --model runs/code_qwen25coder-7b-p7_kd/best \
    --reference code_teacher --metrics chrf
set -f
python train.py --config $CFG --set run_name=code_qwen25coder-1.5b_kd student.init=pretrained \
    student.pretrained=Qwen/Qwen2.5-Coder-1.5B-Instruct student.freeze=[] train.lr=2e-5
set +f
python evaluate.py --config $CFG --name code_qwen25coder-1.5b_kd --model runs/code_qwen25coder-1.5b_kd/best \
    --reference code_teacher --metrics chrf
```

`--metrics chrf` keeps only one text metric (text similarity says little about correct code). Main metrics:
`pass`, `pass_humaneval`, `pass_mbpp`, `code_results` (pass / wrong_answer / runtime_error / syntax_error /
timeout / no_code), `fid_pass_agreement`, `failure.stop_rate`.

## 21. Climate domain: open questions without answers (Researchy Questions, climate subset)

About 5.7k + test-split questions matched by climate keywords; 10% test, 5% validation, rest train (by hash).
No gold answers: the teacher is the reference (fid_* metrics, fid_ppl), plus an optional LLM judge without a reference.

```bash
CFG=configs/climate_llama_L7.yaml

# 1. splits + teacher answers, then check truncation
python prepare_data.py --config $CFG --vllm
python -c "
import json
for s in ('train','validation'):
    r=[json.loads(l) for l in open(f'data/teacher/llama31-8b-it-chat/climate_{s}.jsonl')]
    print(s, len(r), 'stopped:', sum(x['stopped'] for x in r))"

# 2. teacher and zero-shot small model
python evaluate.py --config $CFG --name climate_teacher --model meta-llama/Llama-3.1-8B-Instruct --chat-template
python evaluate.py --config $CFG --name climate_llama32-1b_zs --model meta-llama/Llama-3.2-1B-Instruct \
    --chat-template --reference climate_teacher

# 3a. pruned 7-layer teacher, KD + CE
python train.py --config $CFG --set run_name=climate_llama31-8b-p7_kd
python evaluate.py --config $CFG --name climate_llama31-8b-p7_kd --model runs/climate_llama31-8b-p7_kd/best \
    --reference climate_teacher

# 3b. Llama-3.2-1B, KD + CE (all layers)
set -f
python train.py --config $CFG --set run_name=climate_llama32-1b_kd student.init=pretrained \
    student.pretrained=meta-llama/Llama-3.2-1B-Instruct student.freeze=[] train.lr=2e-5 train.optimizer=adamw
set +f
python evaluate.py --config $CFG --name climate_llama32-1b_kd --model runs/climate_llama32-1b_kd/best \
    --reference climate_teacher

# optional: LLM judge on 100 fixed questions (needs OPENAI_API_KEY), e.g. for the teacher and each student
python evaluate.py --config $CFG --name climate_llama32-1b_kd --rescore --no-ppl --reference climate_teacher \
    --judge openai:gpt-4o-mini --judge-n 100
```

## 22. Climate, long answers (up to 300 words / 512 tokens): same questions and splits as section 21

```bash
CFG=configs/climate_long_llama_L7.yaml
python prepare_data.py --config $CFG --vllm
python -c "
import json
for s in ('train','validation'):
    r=[json.loads(l) for l in open(f'data/teacher/llama31-8b-it-chat/climate_long_{s}.jsonl')]
    print(s, len(r), 'stopped:', sum(x['stopped'] for x in r))"
python evaluate.py --config $CFG --name climate_teacher --model meta-llama/Llama-3.1-8B-Instruct --chat-template
python evaluate.py --config $CFG --name climate_llama32-1b_zs --model meta-llama/Llama-3.2-1B-Instruct \
    --chat-template --reference climate_teacher
python train.py --config $CFG --set run_name=climate_long_llama31-8b-p7_kd
python evaluate.py --config $CFG --name climate_long_llama31-8b-p7_kd --model runs/climate_long_llama31-8b-p7_kd/best \
    --reference climate_teacher
set -f
python train.py --config $CFG --set run_name=climate_long_llama32-1b_kd student.init=pretrained \
    student.pretrained=meta-llama/Llama-3.2-1B-Instruct student.freeze=[] train.lr=2e-5 train.optimizer=adamw
set +f
python evaluate.py --config $CFG --name climate_long_llama32-1b_kd --model runs/climate_long_llama32-1b_kd/best \
    --reference climate_teacher
```

Results land in `eval/<name>/climate_long/` (the teacher and zero-shot names are reused; the dataset folder keeps
them apart from section 21). Compare each student's fid_* and judge score between `climate` and `climate_long`.

## 23. Stitching: low-rank healing layers at the seams (PubMedQA, one-step 32 -> 7)

`train.stitch_rank=64` puts a healing layer h -> h + up(down(h)) (rank 64, about 0.5M parameters, starts as the
identity) in front of every student layer that follows removed teacher layers (6 seams for evenly spaced layers).
`train.stitch_steps=N`: the first N steps train only the healing layers, then all layers; 0 = all together.
Saved as `stitches.pt` next to the checkpoint and re-attached automatically by `load_model`
(evaluate.py, chat.py, energy_benchmark.py without --vllm). Compare with 5c (same run without stitches, 0.633).

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
CFG=configs/pubmedqa_llama_L7_chat.yaml
S="student.num_layers=7 train.ce_weight=0.3 train.max_steps=1500 train.batch_size=2 train.grad_accum=16 train.stitch_rank=64"

# A. healing first (150 steps, only the healing layers), then KD + CE on everything
python train.py --config $CFG --set run_name=pubmedqa_llama31-8b-p7_stitchheal_kd_1500st $S train.stitch_steps=150
# B. healing layers and the rest trained together from the start
python train.py --config $CFG --set run_name=pubmedqa_llama31-8b-p7_stitchjoint_kd_1500st $S train.stitch_steps=0

for n in pubmedqa_llama31-8b-p7_stitchheal_kd_1500st pubmedqa_llama31-8b-p7_stitchjoint_kd_1500st; do
  python evaluate.py --config $CFG --name $n --model runs/$n/best --reference teacher_chat
done
```