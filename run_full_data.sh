#!/bin/bash
# PubMedQA with all ~207k PQA-A training questions (instead of 20k): teacher answers, then 3 trainings.
#   bash run_full_data.sh 2>&1 | tee -a logs/run_full_data.log
# Waits until run_internals.sh has finished (the GPU cannot hold both). Restartable: finished steps are skipped.
# Teacher and zero-shot Llama-3.2-1B need no new run: the test set is the same (teacher_chat, pubmedqa_llama32-1b_zs).
set -f
set -o pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs
CFG=configs/pubmedqa_full_llama_L7_chat.yaml
S="train.ce_weight=0.3 train.gradient_checkpointing=true train.batch_size=2 train.grad_accum=16"

#while pgrep -f run_internals.sh > /dev/null; do echo "$(date +%H:%M) waiting for run_internals.sh"; sleep 600; done

# 1. splits (same test + validation, all remaining PQA-A as train) and teacher answers (vLLM, a few hours)
[ -f data/teacher_full/llama31-8b-it-chat/pubmedqa_train.jsonl ] || python prepare_data.py --config $CFG --vllm || exit 1
cmp -s data/splits/pubmedqa/test.jsonl data/splits_full/pubmedqa/test.jsonl \
  || { echo "test split differs from the 20k setup: stop"; exit 1; }

te() {   # te <run> <extra --set args...>: train + evaluate, skipped if already evaluated
  [ -f eval/$1/pubmedqa/metrics.json ] && { echo "skip $1"; return 0; }
  [ -f runs/$1/summary.json ] || python train.py --config $CFG --set run_name=$1 $S "${@:2}" 2>&1 | tee logs/train_$1.log \
    || { echo "FAILED $1"; return 1; }
  python evaluate.py --config $CFG --name $1 --model runs/$1/best --reference teacher_chat 2>&1 | tee logs/eval_$1.log
  python layer_activity.py --config $CFG --model runs/$1/best
}

# 2. our method, 2 steps: 32 -> 14 layers (CE only), then 14 -> 7 (KD + CE)
P14=runs/pubmedqa-full_llama31-8b-p14_ce_6500st/best
te pubmedqa-full_llama31-8b-p14_ce_6500st     student.num_layers=14 train.objective=sft train.optimizer=adamw_8bit
te pubmedqa-full_llama31-8b-p14-p7_kd_6500st  student.num_layers=7 student.source=$P14
# 3. Llama-3.2-1B with our loss (KD + CE), all layers
te pubmedqa-full_llama32-1b_kd_6500st         student.init=pretrained student.pretrained=meta-llama/Llama-3.2-1B-Instruct \
                                              student.freeze=[] train.lr=2e-5 train.optimizer=adamw
# 4. optional: our 2-step method with the other losses in stage 2 (about 9 h each)
# te pubmedqa-full_llama31-8b-p14-p7_kdrkl_6500st student.num_layers=7 student.source=$P14 train.divergence=rkl
# te pubmedqa-full_llama31-8b-p14-p7_kdskl_6500st student.num_layers=7 student.source=$P14 train.divergence=skl
echo "all done"