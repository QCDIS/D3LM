#!/bin/bash
# All experiments in order, on one machine. Run inside tmux, or:
#   nohup bash run_experiments.sh > logs/all.log 2>&1 &
# Comment out what you don't need. Each training run writes runs/<run_name>/.
set -euo pipefail
set -f                                    # keep [..] in --set values literal
CFG=configs/pubmedqa_llama.yaml
mkdir -p logs

# 1) data: splits + teacher targets
python prepare_data.py --config $CFG --vllm

# 2) teacher reference and no-skill floor on the test set
python evaluate.py --config $CFG --name teacher --model meta-llama/Llama-3.1-8B-Instruct
python evaluate.py --config $CFG --name copy_prompt_floor --model copy_prompt --reference teacher

# 3) depth sweep (choose the depth on best_val_ce in runs/depth_L*/summary.json)
python train.py --config $CFG --layers 1 2 3 4 6 --set run_name=depth

# 4) main runs
python train.py --config $CFG --set run_name=d3lm_L3 student.num_layers=3
python train.py --config $CFG --set run_name=sft_L3 student.num_layers=3 train.objective=sft
python train.py --config $CFG --set run_name=sibling1b_kd student.init=pretrained \
    student.pretrained=meta-llama/Llama-3.2-1B-Instruct "student.freeze=[]" train.lr=2e-5
python train.py --config $CFG --set run_name=sibling1b_sft student.init=pretrained \
    student.pretrained=meta-llama/Llama-3.2-1B-Instruct "student.freeze=[]" train.lr=2e-5 train.objective=sft
python train.py --config $CFG --set run_name=pruned8_kd student.init=teacher_layers student.num_layers=8 \
    "student.freeze=[embeddings]" train.lr=3e-5
python train.py --config $CFG --set run_name=sheared13b_sft student.init=pretrained \
    student.pretrained=princeton-nlp/Sheared-LLaMA-1.3B "student.freeze=[]" train.lr=2e-5 train.objective=sft

# 5) ablations (one seed each)
python train.py --config $CFG --set run_name=abl_rkl train.divergence=rkl
python train.py --config $CFG --set run_name=abl_ce05 train.ce_weight=0.5
python train.py --config $CFG --set run_name=abl_T1 train.temperature=1.0
python train.py --config $CFG --set run_name=abl_gold train.targets=gold

# 6) seeds for the headline rows
for S in 2 3; do
  python train.py --config $CFG --set run_name=d3lm_L3_s$S seed=$S
  python train.py --config $CFG --set run_name=sft_L3_s$S seed=$S train.objective=sft
done

# 7) evaluate every trained model, plus quantised teachers and off-the-shelf small models
for RUN in runs/*/best; do
  NAME=$(basename "$(dirname "$RUN")")
  [ -f "eval/$NAME/pubmedqa/metrics.json" ] || python evaluate.py --config $CFG --name "$NAME" --model "$RUN" --reference teacher
done
python evaluate.py --config $CFG --name teacher_int8 --model meta-llama/Llama-3.1-8B-Instruct --precision int8 --reference teacher
python evaluate.py --config $CFG --name teacher_nf4 --model meta-llama/Llama-3.1-8B-Instruct --precision nf4 --reference teacher
python evaluate.py --config $CFG --name llama32_1b_zeroshot --model meta-llama/Llama-3.2-1B-Instruct --reference teacher

# 8) serving energy: NOTHING else may run on the GPU now
python energy_benchmark.py --config $CFG --name teacher_vllm --model meta-llama/Llama-3.1-8B-Instruct --vllm
python energy_benchmark.py --config $CFG --name teacher_awq_vllm --vllm --quantization awq \
    --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
python energy_benchmark.py --config $CFG --name d3lm_L3_vllm --model runs/d3lm_L3/best --vllm
python energy_benchmark.py --config $CFG --name sibling_kd_vllm --model runs/sibling1b_kd/best --vllm
python energy_benchmark.py --config $CFG --name teacher_hf --model meta-llama/Llama-3.1-8B-Instruct
python energy_benchmark.py --config $CFG --name d3lm_L3_hf --model runs/d3lm_L3/best

# 9) break-even
python break_even.py --spec configs/break_even.yaml
