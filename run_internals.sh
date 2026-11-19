#!/bin/bash
# Top 3 methods x 3 datasets, 1500 steps, then a layer-activity picture of every model.
#   DRY=1 bash run_internals.sh                              # print the commands only
#   bash run_internals.sh 2>&1 | tee logs/run_internals.log  # run (use tmux / nohup)
# Methods: (A) small pretrained student, KD + CE, all layers
#          (B) teacher pruned to 7 layers in one step, KD + CE + hidden-state loss
#          (C) gradual pruning: 14 layers CE only, then 7 layers KD + CE
# A finished step is skipped: training if runs/<run>/summary.json exists, evaluation if its metrics.json exists,
# the picture if its png exists. So the script can be restarted after a crash.
# Memory: batch 2 x accum 16 (effective 32), gradient checkpointing, 8-bit AdamW for every student except the
# 1B Llama (fp32 AdamW, ~20 GB). If a training fails anyway, it is retried once with batch 1 x accum 32 (same
# effective batch, so results stay comparable).
set -f
set -o pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs
run() { echo "+ $*"; [ -n "$DRY" ] || "$@"; }

COMMON="train.max_steps=1500 train.ce_weight=0.3 train.gradient_checkpointing=true train.batch_size=2 train.grad_accum=16"
SMALL="student.init=pretrained student.freeze=[] train.lr=2e-5"

te() {   # te <config> <dataset> <reference> <run> <extra --set args...>
  local cfg=$1 ds=$2 ref=$3 n=$4; shift 4
  local metrics=""; [ $ds = code ] && metrics="--metrics chrf"
  if [ ! -f runs/$n/summary.json ]; then
    run python train.py --config $cfg --set run_name=$n $COMMON "$@" 2>&1 | tee logs/train_$n.log \
      || run python train.py --config $cfg --set run_name=$n $COMMON "$@" train.batch_size=1 train.grad_accum=32 \
           2>&1 | tee logs/train_${n}_retry.log \
      || { echo "FAILED $n"; return 1; }
  fi
  [ -f eval/$n/$ds/metrics.json ] || run python evaluate.py --config $cfg --name $n --model runs/$n/best \
      --reference $ref $metrics 2>&1 | tee logs/eval_$n.log
  [ -f results/layer_activity/${ds}__$n.png ] || run python layer_activity.py --config $cfg --model runs/$n/best
}

three() {   # three <config> <dataset> <reference> <teacher HF id> <teacher tag> <small tag> <small HF id> <small optimizer>
  local cfg=$1 ds=$2 ref=$3 tid=$4 t=$5 st=$6 sid=$7 opt=$8
  echo "=== $ds  $(df -h . | tail -1 | awk '{print $4}') free"
  te $cfg $ds $ref ${ds}_${st}_kd_1500st      $SMALL student.pretrained=$sid train.optimizer=$opt     # A
  te $cfg $ds $ref ${ds}_${t}-p7_kdhid_1500st student.num_layers=7 train.hidden_weight=1.0           # B
  te $cfg $ds $ref ${ds}_${t}-p14_ce_1500st   student.num_layers=14 train.objective=sft \
                                              train.optimizer=adamw_8bit                              # C stage 1
  te $cfg $ds $ref ${ds}_${t}-p14-p7_kd_1500st student.num_layers=7 \
                                              student.source=runs/${ds}_${t}-p14_ce_1500st/best        # C stage 2
  # stage 1 is only a stepping stone: free its ~15 GB once stage 2 is done
  [ -f runs/${ds}_${t}-p14-p7_kd_1500st/summary.json ] && run rm -rf runs/${ds}_${t}-p14_ce_1500st/best
  # teacher picture, for comparison (chat template, as the teacher answered)
  [ -f results/layer_activity/${ds}__$(basename $tid).png ] \
      || run python layer_activity.py --config $cfg
}

# preflight: teacher answers and teacher references must exist
for f in data/teacher/llama31-8b-it-chat/pubmedqa_train.jsonl data/teacher/llama31-8b-it-chat/climate_train.jsonl \
         data/teacher/qwen25-coder-7b-it-chat/code_train.jsonl eval/teacher_chat eval/climate_teacher eval/code_teacher; do
  [ -e $f ] || { echo "missing $f (see COMMANDS.md sections 1, 2, 20, 21)"; exit 1; }
done

three configs/pubmedqa_llama_L7_chat.yaml pubmedqa teacher_chat meta-llama/Llama-3.1-8B-Instruct llama31-8b llama32-1b meta-llama/Llama-3.2-1B-Instruct adamw
three configs/climate_llama_L7.yaml       climate  climate_teacher meta-llama/Llama-3.1-8B-Instruct llama31-8b llama32-1b meta-llama/Llama-3.2-1B-Instruct adamw
three configs/code_qwen_L7.yaml           code     code_teacher Qwen/Qwen2.5-Coder-7B-Instruct qwen25coder-7b qwen25coder-1.5b Qwen/Qwen2.5-Coder-1.5B-Instruct adamw_8bit
echo "all done"