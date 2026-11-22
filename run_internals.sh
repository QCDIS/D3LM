#!/bin/bash
# Distillation comparison on all 3 datasets (pubmedqa, climate, code), 1500 steps per training.
# Every model is evaluated and gets a layer-activity picture.
#   DRY=1 bash run_internals.sh                               # print the commands + how many trainings are left
#   bash run_internals.sh 2>&1 | tee -a logs/run_internals.log
#   KEEP=1 bash run_internals.sh ...                          # keep checkpoints (default: deleted after eval + picture)
#   SKIP="climate_* *_sftgold_*" bash run_internals.sh ...    # leave out runs whose name matches a pattern
#
#   0  no training: zero-shot small model, teacher quantized to int8 and nf4
#   1  7-layer pruned teacher: SFT (CE only), KD+CE with fkl, rkl, skl
#   2  small pretrained student (Llama-3.2-1B; Qwen2.5-Coder-1.5B for code): KD+CE fkl
#   3  7 layers: hidden-state loss only, KD+CE+hidden; gradual 14 -> 7 (KD+CE, KD+CE+hidden) only where the
#      stage-1 checkpoint runs/<dataset>_<teacher>-p14_ce_1500st/best already exists
#   4  SFT on the gold answers instead of the teacher's (pubmedqa, code; climate has no gold answers)
#
# Settings for every training: 1500 steps, ce_weight 0.3, batch 2 x accum 16, gradient checkpointing, 8-bit AdamW
# (fp32 AdamW for Llama-3.2-1B). A training that fails is retried once with batch 1 x accum 32 (same effective
# batch). Finished steps are skipped (training: summary.json or an evaluation exists; evaluation: metrics.json;
# picture: png), so the script can be restarted at any time. Code evaluations made before the HumanEval fix are
# redone if the model still exists. It stops when less than 40 GB of disk is free.
set -f
set -o pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs
run() { echo "+ $*"; [ -n "$DRY" ] || "$@"; }
TODO=0

COMMON="train.max_steps=1500 train.ce_weight=0.3 train.gradient_checkpointing=true train.batch_size=2 train.grad_accum=16"
SMALL="student.init=pretrained student.freeze=[] train.lr=2e-5"
P7="student.init=teacher_layers student.num_layers=7"

skipped() { local s; for s in $SKIP; do [[ $1 == $s ]] && return 0; done; return 1; }

stale_code_eval() {   # true if a code evaluation was made with the old HumanEval prompt (empty answers)
  python -c "
import json, sys
m = json.load(open('$1'))
sys.exit(0 if m.get('pass_humaneval', 1) == 0 and m['code_results'].get('no_code', 0) > 0.2 else 1)"
}

ev() {   # ev <config> <dataset> <reference> <name> <evaluate.py args...>: evaluation only
  local cfg=$1 ds=$2 ref=$3 n=$4; shift 4
  skipped $n && { echo "skip (SKIP) $n"; return 0; }
  local m=eval/$n/$ds/metrics.json metrics=""; [ $ds = code ] && metrics="--metrics chrf"
  [ -f $m ] && [ $ds = code ] && stale_code_eval $m && run rm -f $m
  [ -f $m ] || run python evaluate.py --config $cfg --name $n "$@" --reference $ref $metrics 2>&1 | tee logs/eval_$n.log
  return 0
}

te() {   # te <config> <dataset> <reference> <run> <extra --set args...>: train, evaluate, picture
  local cfg=$1 ds=$2 ref=$3 n=$4; shift 4
  skipped $n && { echo "skip (SKIP) $n"; return 0; }
  local m=eval/$n/$ds/metrics.json metrics="" new=""; [ $ds = code ] && metrics="--metrics chrf"
  [ -f $m ] && [ $ds = code ] && [ -d runs/$n/best ] && stale_code_eval $m && run rm -f $m
  if [ ! -f runs/$n/summary.json ] && [ ! -f $m ]; then
    TODO=$((TODO + 1)); new=1
    if [ -z "$DRY" ] && [ $(df -BG --output=avail . | tail -1 | tr -dc 0-9) -lt 40 ]; then
      echo "STOP: less than 40 GB free disk before $n (free space, then rerun the script)"; exit 1
    fi
    run python train.py --config $cfg --set run_name=$n $COMMON "$@" 2>&1 | tee logs/train_$n.log \
      || run python train.py --config $cfg --set run_name=$n $COMMON "$@" train.batch_size=1 train.grad_accum=32 \
           2>&1 | tee logs/train_${n}_retry.log \
      || { echo "FAILED $n"; return 1; }
  fi
  if [ -d runs/$n/best ] || { [ -n "$DRY" ] && [ -n "$new" ]; }; then
    [ -f $m ] || run python evaluate.py --config $cfg --name $n --model runs/$n/best --reference $ref $metrics \
        2>&1 | tee logs/eval_$n.log
    [ -f results/layer_activity/${ds}__$n.png ] || run python layer_activity.py --config $cfg --model runs/$n/best
    # only checkpoints trained by this run of the script are deleted
    [ -n "$new" ] && [ -z "$KEEP" ] && [ -f $m ] && [ -f results/layer_activity/${ds}__$n.png ] \
        && run rm -rf runs/$n/best
  fi
  return 0
}

# datasets: config | dataset | teacher reference | teacher HF id | teacher tag | small tag | small HF id | optimizer
DATASETS=(
  "configs/pubmedqa_llama_L7_chat.yaml pubmedqa teacher_chat meta-llama/Llama-3.1-8B-Instruct llama31-8b llama32-1b meta-llama/Llama-3.2-1B-Instruct adamw"
  "configs/climate_llama_L7.yaml climate climate_teacher meta-llama/Llama-3.1-8B-Instruct llama31-8b llama32-1b meta-llama/Llama-3.2-1B-Instruct adamw"
  "configs/code_qwen_L7.yaml code code_teacher Qwen/Qwen2.5-Coder-7B-Instruct qwen25coder-7b qwen25coder-1.5b Qwen/Qwen2.5-Coder-1.5B-Instruct adamw_8bit"
)

tier0() {   # no training
  local cfg=$1 ds=$2 ref=$3 tid=$4 t=$5 st=$6 sid=$7
  ev $cfg $ds $ref ${ds}_${st}_zs  --model $sid --chat-template
  ev $cfg $ds $ref ${ds}_${t}_int8 --model $tid --chat-template --precision int8
  ev $cfg $ds $ref ${ds}_${t}_nf4  --model $tid --chat-template --precision nf4
}

tier1() {   # losses, 7-layer pruned teacher
  local cfg=$1 ds=$2 ref=$3 tid=$4 t=$5
  te $cfg $ds $ref ${ds}_${t}-p7_sft_1500st   $P7 train.objective=sft
  te $cfg $ds $ref ${ds}_${t}-p7_kd_1500st    $P7
  te $cfg $ds $ref ${ds}_${t}-p7_kdrkl_1500st $P7 train.divergence=rkl
  te $cfg $ds $ref ${ds}_${t}-p7_kdskl_1500st $P7 train.divergence=skl
}

tier2() {   # small pretrained student, all layers
  local cfg=$1 ds=$2 ref=$3 tid=$4 t=$5 st=$6 sid=$7 opt=$8
  te $cfg $ds $ref ${ds}_${st}_kd_1500st $SMALL student.pretrained=$sid train.optimizer=$opt
}

tier3() {   # hidden-state loss; gradual 14 -> 7 only if the stage-1 checkpoint exists
  local cfg=$1 ds=$2 ref=$3 tid=$4 t=$5
  te $cfg $ds $ref ${ds}_${t}-p7_hid_1500st   $P7 train.objective=hidden
  te $cfg $ds $ref ${ds}_${t}-p7_kdhid_1500st $P7 train.hidden_weight=1.0
  local p14=runs/${ds}_${t}-p14_ce_1500st/best a=${ds}_${t}-p14-p7_kd_1500st b=${ds}_${t}-p14-p7_kdhid_1500st
  if [ -d $p14 ]; then
    te $cfg $ds $ref $a $P7 student.source=$p14
    te $cfg $ds $ref $b $P7 student.source=$p14 train.hidden_weight=1.0
  else
    echo "gradual 14 -> 7 for $ds: no stage-1 checkpoint $p14, not trained"
  fi
  return 0
}

tier4() {   # SFT on gold answers (climate has none)
  local cfg=$1 ds=$2 ref=$3 tid=$4 t=$5 st=$6 sid=$7 opt=$8
  [ $ds = climate ] && return 0
  te $cfg $ds $ref ${ds}_${t}-p7_sftgold_1500st $P7 train.objective=sft train.targets=gold
  te $cfg $ds $ref ${ds}_${st}_sftgold_1500st   $SMALL student.pretrained=$sid train.optimizer=$opt \
                                                train.objective=sft train.targets=gold
}

# ------------------------------------------------------------------ preflight
for f in data/teacher/llama31-8b-it-chat/pubmedqa_train.jsonl data/teacher/llama31-8b-it-chat/climate_train.jsonl \
         data/teacher/qwen25-coder-7b-it-chat/code_train.jsonl data/splits/code/test.jsonl \
         eval/teacher_chat/pubmedqa/metrics.json eval/climate_teacher/climate/metrics.json eval/code_teacher/code/metrics.json; do
  [ -e $f ] || { echo "missing $f (see COMMANDS.md sections 1, 2, 20, 21)"; exit 1; }
done
grep -q "torch.cuda.empty_cache()" train.py || { echo "old train.py (no memory fix before validation): update it"; exit 1; }
grep -q "Write the complete function" common.py || { echo "old common.py (HumanEval prompt bug): update it"; exit 1; }
python -c "
import json, sys
bad = sum(json.loads(l)['prompt'].endswith('\`\`\`') for l in open('data/splits/code/test.jsonl'))
sys.exit(f'{bad} HumanEval test prompts still end with a code fence: run COMMANDS.md section 20b step 1' if bad else 0)" || exit 1
stale_code_eval eval/code_teacher/code/metrics.json \
  && { echo "code_teacher was evaluated before the HumanEval fix: run COMMANDS.md section 20b step 2"; exit 1; }
python -c "import bitsandbytes" || { echo "bitsandbytes missing (8-bit AdamW, int8/nf4)"; exit 1; }
echo "preflight ok, $(df -h --output=avail . | tail -1 | tr -d ' ') free disk"

for tier in tier0 tier1 tier2 tier3 tier4; do
  for d in "${DATASETS[@]}"; do
    set -- $d
    echo "=== $tier $2"
    $tier "$@"
  done
done
# teacher pictures, for comparison
for d in "${DATASETS[@]}"; do
  set -- $d
  [ -f results/layer_activity/${2}__$(basename $4).png ] || run python layer_activity.py --config $1
done
[ -n "$DRY" ] && echo "trainings still to run: $TODO"
grep -h FAILED logs/run_internals.log 2>/dev/null | sort -u
echo "all done"