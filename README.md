# D3LM: domain-distilled small LMs (CCGrid'27 experiments)

Six scripts and a few YAML configs. Run everything from this folder, on one machine (no scheduler needed).

| File | Purpose |
|---|---|
| `common.py` | shared code: config loading, datasets and splits, model loading, student building, tokenisation, generation, GPU energy meter |
| `prepare_data.py` | step 1: train/val/test splits (saved once, no overlap) + teacher training targets |
| `train.py` | step 2: KD (D3LM) or SFT in one loop; divergences fkl / rkl / skl / srkl; `--layers` for a depth sweep |
| `evaluate.py` | step 3: scores continuations only: task labels, long answers, fidelity to the teacher, perplexity / bits per character, floors, optional LLM judge; `--rescore` re-scores saved predictions |
| `energy_benchmark.py` | step 4: latency / throughput / J per request (HF batch sizes, or vLLM under load) |
| `break_even.py` | step 5: lifecycle energy (data generation + depth search + training + serving), break-even |
| `configs/` | `base.yaml` (all settings, commented), `pubmedqa_llama.yaml`, `pubmedqa_qwen.yaml`, `wikitext_squad_llama.yaml`, `break_even.yaml` |
| `COMMANDS.md` | every experiment in order, as copy-paste commands (data, training, baselines, sweeps, evaluation, energy) |
| `qualitative.py` | 10 random test cases with each model's saved answer, side by side |
| `chat.py` | ask a model questions in the terminal |
| `test_smoke.py` | offline check with tiny models: `python -m pytest -q test_smoke.py` |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # must match the driver: `nvidia-smi` shows "CUDA Version"
pip install -r requirements.txt          # includes sentence-transformers and bert-score for evaluate.py
pip install openai google-genai          # only for the LLM judge (needs OPENAI_API_KEY / GEMINI_API_KEY)
huggingface-cli login                    # Llama models are gated
python -m pytest -q test_smoke.py        # 5 offline tests, CPU only
```
If PyTorch reports "NVIDIA driver too old", reinstall it from the index that matches the driver (above), then install vLLM without replacing it (`pip install vllm --no-deps`).

## First run
```bash
CFG=configs/pubmedqa_llama.yaml          # teacher: meta-llama/Llama-3.1-8B-Instruct
python prepare_data.py --config $CFG --vllm
python evaluate.py --config $CFG --name teacher --model meta-llama/Llama-3.1-8B-Instruct
python train.py    --config $CFG --set run_name=d3lm_L3
python train.py    --config $CFG --set run_name=sft_L3 train.objective=sft
python evaluate.py --config $CFG --name d3lm_L3 --model runs/d3lm_L3/best --reference teacher
python evaluate.py --config $CFG --name sft_L3  --model runs/sft_L3/best  --reference teacher
```
Any setting can be overridden with `--set key=value` (dotted keys, e.g. `student.num_layers=4`). Long runs: start them in `tmux`, or with `nohup ... > logs/x.log 2>&1 &`.

## Step 1: `prepare_data.py`
- Builds `data/splits/<dataset>/{train,validation,test}.jsonl` once. Test is sampled first, then validation, then train; anything that shares a prompt or a source (article, PubMed id) with an earlier split is dropped. `manifest.json` stores a sha256 per file, and every later step refuses to run if a file changed.
- PubMedQA: test = the 1,000 expert-labelled PQA-L questions (yes / no / maybe); train and validation = PQA-A (artificial, yes / no only). Prompt = instruction + abstract (`context.contexts`) + question + `Answer:`. `long_answer` and `final_decision` are kept as gold references.
- The teacher (`teacher:` in the config) answers every train and validation prompt (greedy, `max_new_tokens`). The answers go to `data/teacher/<teacher_tag>/<dataset>_{train,validation}.jsonl` as `{id, completion, stopped}`; the energy used goes to `manifest.json` in the same folder (data-generation cost for `break_even.py`).
- **Known issue:** without a chat template the instruct teacher often ignores the requested format, rambles and is cut off at 128 tokens (`"stopped": false`). Check the share of truncated targets before large runs and clean or regenerate them if it is high (keep text up to the first blank line, or generate with the chat template).

## Step 2: `train.py`
One training example = prompt + teacher completion + EOS. Only completion tokens and the EOS carry loss; the prompt is context. Each step:
1. student forward (fp32 master weights, bf16 autocast) and teacher forward (no gradients; skipped for SFT);
2. loss = `(1 - ce_weight) * T^2 * D(teacher, student) + ce_weight * CE`, over completion tokens only;
3. gradient accumulation, clipping, AdamW with warm-up + cosine schedule;
4. every `eval_every` steps: validation `val_ce` (selection metric, comparable across KD and SFT) and `val_kl`; the best checkpoint is saved to `runs/<run_name>/best/`; early stopping after `patience` evaluations without improvement.

| Setting | Values |
|---|---|
| `student.init` | `random` (D3LM: teacher architecture with `num_layers` fresh blocks, embeddings/head/norm copied), `teacher_layers` (blocks copied from the teacher), `pretrained` (e.g. `meta-llama/Llama-3.2-1B-Instruct`) |
| `train.objective` | `kd`, `sft` (same loop without the KD term; the teacher is not run) |
| `train.divergence` | `fkl` = KL(p‖q) (original D3LM), `rkl` = KL(q‖p), `skl` = KL(p‖αp+(1−α)q) and `srkl` = KL(q‖(1−α)p+αq) (skew KL, DistiLLM, ICML'24), with `train.skew_alpha` (default 0.1) |
| `train.targets` | `teacher` (default), `gold` (`"<label>. <long_answer>"`, ablation) |

`val_kl` reports whichever divergence a run uses, so compare runs with `best_val_ce` and test metrics. Outputs: `runs/<run_name>/{config.json, log.jsonl, best/, summary.json}`; `summary.json` holds steps, best `val_ce`, parameter counts, wall time and GPU energy (training cost for `break_even.py`).

## Step 3: `evaluate.py`
Generates greedy answers for the test split (identical decoding for every model, continuation only, never the prompt) and writes `eval/<name>/<dataset>/{predictions.jsonl, metrics.json}`. Accuracy and the text metrics have bootstrap 95% CIs; perplexity and bits per character are corpus-level values.

| Group | Compared with | Metrics |
|---|---|---|
| task | gold label / span | PubMedQA accuracy + macro-F1 (first yes/no/maybe), SQuAD EM/F1 |
| `gold_*` | gold reference text: PubMedQA `long_answer` (vs the justification after the label), WikiText continuation, `gold` of own jsonl datasets | BLEU, ROUGE-1, ROUGE-L, chrF, cosine similarity (`all-MiniLM-L6-v2`), BERTScore-F1 (`roberta-large`) |
| `fid_*` | the reference model's answer (`--reference teacher`) | the same six metrics |
| `*_floor_shuffled` | shuffled references | the same metrics: what an answer with no skill scores |
| `gold_ppl`, `gold_bpc` | likelihood of the gold answer given the prompt (PubMedQA: `"<label>. <long_answer>"`) | perplexity (same tokenizer only) and bits per character (any tokenizer) |
| `fid_ppl`, `fid_bpc` | likelihood of the teacher's answer given the prompt | test-set counterpart of `val_ce` |
| `judge_<provider>` | gold reference + abstract | LLM judge, 0–10, on a fixed random subset (`--judge-n`, same items for every model) |
| `failure` | – | empty rate, EOS stop rate, degenerate repetition, mean length |

Useful options:
```bash
python evaluate.py --config $CFG --name floor --model copy_prompt --reference teacher           # no-skill baseline
python evaluate.py --config $CFG --name teacher_int8 --model meta-llama/Llama-3.1-8B-Instruct --precision int8 --reference teacher
python evaluate.py --config $CFG --name teacher --rescore --metrics-device cpu                   # new metrics, no GPU, no regeneration
python evaluate.py --config $CFG --name teacher --rescore --model meta-llama/Llama-3.1-8B-Instruct   # add perplexity without regenerating
python evaluate.py --config $CFG --name d3lm_L3 --rescore --reference teacher --judge openai:gpt-4o-mini --judge-n 100
```
`--metrics` selects text metrics (scoring models: `all-MiniLM-L6-v2` for cosine, `roberta-large` for BERTScore, set at the top of `evaluate.py`), `--no-ppl` skips perplexity, `--chat-template` wraps prompts in the model's chat template. Evaluate the teacher first (without `--reference`); every other model uses `--reference teacher`. Judge answers are cached in `judge_<provider>.json`.

### Rescoring saved predictions (`--rescore`)
Generation is the expensive part of evaluation. `--rescore` reuses the answers already saved in `eval/<name>/<dataset>/predictions.jsonl` and only recomputes the scores, so you can add metrics, a judge or a new reference model without generating again.

What happens with `--rescore`:
1. the test split is loaded and checked against its sha256, exactly as in a normal run;
2. the saved answer, token count and stop flag of every test item are read from `predictions.jsonl` (matched by `id`);
3. all scores are recomputed from those answers: task scores, `gold_*` and `fid_*` text metrics, floors, failure statistics, and the judge if `--judge` is given;
4. `predictions.jsonl` and `metrics.json` are rewritten; `generation_energy` is copied from the previous `metrics.json`.

| Command | Loads a model? | Recomputes |
|---|---|---|
| `--rescore` | no (the scoring models for cosine / BERTScore only) | task, text metrics, floors, failure stats |
| `--rescore --model <same model>` | yes | the above + `gold_ppl/bpc` and `fid_ppl/bpc` |
| `--rescore --reference teacher` | no | the above + `fid_*` against the teacher's saved answers |
| `--rescore --judge openai:<model> --judge-n 100` | no | the above + judge scores (cached; only new items are sent) |
| `--rescore --metrics bleu rougeL --metrics-device cpu` | no | only the listed text metrics, on the CPU |

Typical uses:
```bash
# new metrics for the teacher while a training run occupies the GPU (CPU only, no energy interference)
python evaluate.py --config $CFG --name teacher --rescore --metrics-device cpu
# add perplexity to an existing evaluation without regenerating (loads the model on the GPU)
python evaluate.py --config $CFG --name d3lm_L3 --rescore --model runs/d3lm_L3/best --reference teacher
# judge several models on the same 100 items
for M in teacher d3lm_L3 sft_L3; do
  python evaluate.py --config $CFG --name $M --rescore --reference teacher --judge openai:gpt-4o-mini --judge-n 100
done
```

Things to know:
- `--rescore` needs a previous normal run of the same `--name` (its `predictions.jsonl` must contain every test id); it never generates.
- The answers are not regenerated, so decoding changes (e.g. a different `max_new_tokens`) need a normal run instead.
- `metrics.json` is replaced, not merged: perplexity is only kept if the rescoring run also passes `--model`, and `fid_*` only if it also passes `--reference`. Give the full set of options you want each time.
- `--reference <name>` needs that model's `predictions.jsonl` for the same dataset (evaluate the teacher first).
- Run CPU rescoring (`--metrics-device cpu`, no `--model`) next to a training run; anything that loads a model on the GPU should wait until training has finished, or it disturbs the training energy measurement.

## Hyper-parameter sweep
`python hparam_sweep.py configs/pubmedqa_llama_L7.yaml` trains and evaluates each variant in `VARIANTS`
(cleaned targets, ce_weight, temperature, divergence, trainable head, layer choice) and appends one row per run
to `results/hparam_sweep.csv`. Add or edit variants directly in the dict at the top of the script.

## Steps 4–5: energy and break-even
```bash
python energy_benchmark.py --config $CFG --name d3lm_L3_vllm --model runs/d3lm_L3/best --vllm   # nothing else on the GPU
python break_even.py --spec configs/break_even.yaml
```
`energy_benchmark.py` measures idle power, warms up, forces exactly `max_new_tokens` per request, and reports J/request (total and idle-subtracted), throughput and latency percentiles; with `--vllm` also per request rate and the highest rate that meets the latency SLO. `break_even.py` combines data-generation, depth-search and training energy with energy per request into the number of requests (and days) after which a student pays for itself.

## Fixes relative to the eScience code
- metrics use the generated continuation only (the old code scored prompt + continuation, which inflated BLEU/ROUGE/BERTScore)
- KD and SFT share one training loop and one evaluation (the old SFT used a different trainer and evaluator)
- QA prompts include the question; evaluation uses gold labels and long answers, not only similarity to the teacher
- splits are created once, checked for train/test overlap and hashed
- EOS is a training target (the old mask dropped it because pad = eos)
- decoding is greedy and identical for all models
- energy uses the NVML energy counter over long windows, with idle power subtracted
- students keep the teacher's own architecture class (the old code made every student a Llama)