#!/bin/bash
# Small models from other families (Qwen2.5-1.5B, SmolLM2-1.7B, Gemma-3-1B) on all 3 datasets.
#   DRY=1 bash run_small_models.sh                                # print the commands only
#   bash run_small_models.sh 2>&1 | tee -a logs/run_small_models.log
# For every dataset and model: zero-shot evaluation, training on the teacher's answers, evaluation, layer picture.
#   - CE + KL wherever the student shares the teacher's tokenizer: only Qwen2.5-1.5B on code (Qwen2.5-Coder-7B
#     teacher), with ce_weight 0.3 and temperature 2, like the other KD students.
#   - CE only (SFT) everywhere else: a different tokenizer makes token-level KL impossible.
# PubMedQA uses the full PQA-A training set (~207k teacher answers) for 6500 steps; climate and code use 1500 steps.
# 9 trainings, one per model and dataset. Finished steps are skipped; a failed training is retried once with batch 1 x accum 32.
set -f
set -o pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs
run() { echo "+ $*"; [ -n "$DRY" ] || "$@"; }

COMMON="student.init=pretrained student.freeze=[] train.lr=2e-5 train.gradient_checkpointing=true"
CE="train.objective=sft train.optimizer=adamw"
KD="train.objective=kd train.ce_weight=0.3 train.optimizer=adamw_8bit"     # 8-bit: the teacher is loaded too

MODELS=("qwen25-1.5b Qwen/Qwen2.5-1.5B-Instruct"
        "smollm2-1.7b HuggingFaceTB/SmolLM2-1.7B-Instruct"
        "gemma3-1b google/gemma-3-1b-it")
# config | dataset | teacher reference | run-name prefix | training steps
DATASETS=("configs/pubmedqa_full_llama_L7_chat.yaml pubmedqa teacher_chat pubmedqa-full 6500"
          "configs/climate_llama_L7.yaml climate climate_teacher climate 1500"
          "configs/code_qwen_L7.yaml code code_teacher code 1500")

te() {   # te <config> <dataset> <reference> <run> <extra --set args...>: train, evaluate, picture
  local cfg=$1 ds=$2 ref=$3 n=$4; shift 4
  local metrics=""; [ $ds = code ] && metrics="--metrics chrf"
  if [ ! -f runs/$n/summary.json ] && [ ! -f eval/$n/$ds/metrics.json ]; then
    run python train.py --config $cfg --set run_name=$n "$@" train.batch_size=2 train.grad_accum=16 \
        2>&1 | tee logs/train_$n.log \
      || run python train.py --config $cfg --set run_name=$n "$@" train.batch_size=1 train.grad_accum=32 \
           2>&1 | tee logs/train_${n}_retry.log \
      || { echo "FAILED $n"; return 0; }
  fi
  if [ -d runs/$n/best ] || [ -n "$DRY" ]; then
    [ -f eval/$n/$ds/metrics.json ] || run python evaluate.py --config $cfg --name $n --model runs/$n/best \
        --reference $ref $metrics 2>&1 | tee logs/eval_$n.log
    [ -f results/layer_activity/${ds}__$n.png ] || run python layer_activity.py --config $cfg --model runs/$n/best
  fi
  return 0
}

for d in "${DATASETS[@]}"; do
  set -- $d; cfg=$1; ds=$2; ref=$3; prefix=$4; steps=$5
  metrics=""; [ $ds = code ] && metrics="--metrics chrf"
  for m in "${MODELS[@]}"; do
    set -- $m; tag=$1; hf=$2
    echo "=== $ds $tag"
    # zero-shot, with the chat template (does not depend on the training data)
    zs=${ds}_${tag}_zs
    [ -f eval/$zs/$ds/metrics.json ] || run python evaluate.py --config $cfg --name $zs --model $hf --chat-template \
        --reference $ref $metrics 2>&1 | tee logs/eval_$zs.log
    # one training: CE + KL where the tokenizers match (Qwen2.5-1.5B with the Qwen2.5-Coder teacher), else CE only
    if [ $ds = code ] && [ $tag = qwen25-1.5b ]; then
      te $cfg $ds $ref ${prefix}_${tag}_kd_${steps}st $COMMON $KD student.pretrained=$hf train.max_steps=$steps
    else
      te $cfg $ds $ref ${prefix}_${tag}_sft_${steps}st $COMMON $CE student.pretrained=$hf train.max_steps=$steps
    fi
  done
done
grep -h FAILED logs/run_small_models.log 2>/dev/null | sort -u
echo "all done"