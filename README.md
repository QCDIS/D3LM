# D3LM: domain-distilled small LMs (CCGrid'27 experiments)

Six scripts and a few YAML configs. Run everything from this folder, on one machine.

| File | Purpose |
|---|---|
| `common.py` | shared code: config loading, datasets and splits, model loading, student building, tokenisation, generation, GPU energy meter |
| `prepare_data.py` | step 1: train/val/test splits (saved once, no overlap) + teacher training targets |
| `train.py` | step 2: KD (D3LM) or SFT in one loop; `--layers` for a depth sweep |
| `evaluate.py` | step 3: scores continuations only, vs gold labels and vs the teacher; CIs, floor, failure stats |
| `energy_benchmark.py` | step 4: latency / throughput / J per request (HF batch sizes, or vLLM under load) |
| `break_even.py` | step 5: lifecycle energy (data generation + depth search + training + serving), break-even |
| `configs/` | `base.yaml` (all settings, commented) + one file per experiment + `break_even.yaml` |
| `run_experiments.sh` | every experiment in order |
| `test_smoke.py` | offline check with tiny models: `python -m pytest -q test_smoke.py` |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your NVIDIA driver (nvidia-smi)
pip install -r requirements.txt
huggingface-cli login
```

## First run
```bash
CFG=configs/pubmedqa_llama.yaml
python prepare_data.py --config $CFG --vllm
python evaluate.py --config $CFG --name teacher --model meta-llama/Llama-3.1-8B-Instruct
python train.py    --config $CFG --set run_name=d3lm_L3
python train.py    --config $CFG --set run_name=sft_L3 train.objective=sft
python evaluate.py --config $CFG --name d3lm_L3 --model runs/d3lm_L3/best --reference teacher
python evaluate.py --config $CFG --name sft_L3  --model runs/sft_L3/best  --reference teacher
```
Any setting can be overridden with `--set key=value` (dotted keys, e.g. `student.num_layers=4`).

## Fixes relative to the eScience code
- metrics use the generated continuation only (the old code scored prompt + continuation)
- KD and SFT share one training loop and one evaluation (the old SFT used a different trainer and evaluator)
- SQuAD/PubMedQA prompts include the question; evaluation uses gold labels (EM/F1, accuracy)
- splits are created once, checked for train/test overlap and hashed
- EOS is a training target (the old mask dropped it because pad = eos)
- decoding is greedy and identical for all models
- energy uses the NVML energy counter over long windows, with idle power subtracted
- students keep the teacher's own architecture class (the old code made every student a Llama)
