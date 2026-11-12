#!/bin/bash
# Student screening, same KD + CE training as the main student (ce_weight 0.3).
#   bash screen_students.sh 2>&1 | tee logs/screen_students.log
# Small models: all layers, fp32. 8B models: pruned to 7 layers, current L7 settings.
# Non-Llama models have a different tokenizer, so the KD term is impossible: they train on CE only
# (train.objective=sft) on the same teacher answers.
set -f
mkdir -p logs
CFG=configs/pubmedqa_llama_L7_chat.yaml
SMALL="student.init=pretrained student.freeze=[] train.lr=2e-5 train.optimizer=adamw train.ce_weight=0.3"
BIG="student.init=teacher_layers student.num_layers=7 train.ce_weight=0.3"

train_eval() {
  python train.py --config $CFG --set run_name=$1 "${@:2}"
  python evaluate.py --config $CFG --name $1 --model runs/$1/best --reference teacher_chat
}

# small models, all layers
train_eval pubmedqa_llama32-1b_kd    student.pretrained=meta-llama/Llama-3.2-1B-Instruct    $SMALL
train_eval pubmedqa_llama32-3b_kd    student.pretrained=meta-llama/Llama-3.2-3B-Instruct    $SMALL
train_eval pubmedqa_qwen25-1.5b_ce   student.pretrained=Qwen/Qwen2.5-1.5B-Instruct          $SMALL train.objective=sft
train_eval pubmedqa_gemma3-1b_ce     student.pretrained=google/gemma-3-1b-it                $SMALL train.objective=sft
train_eval pubmedqa_smollm2-1.7b_ce  student.pretrained=HuggingFaceTB/SmolLM2-1.7B-Instruct $SMALL train.objective=sft
train_eval pubmedqa_granite33-2b_ce  student.pretrained=ibm-granite/granite-3.3-2b-instruct $SMALL train.objective=sft

# 8B models, 7 layers
train_eval pubmedqa_openbiollm-8b-p7_kd student.source=aaditya/Llama3-OpenBioLLM-8B          $BIG
train_eval pubmedqa_qwen25-7b-p7_ce     student.source=Qwen/Qwen2.5-7B-Instruct              $BIG train.objective=sft
train_eval pubmedqa_granite33-8b-p7_ce  student.source=ibm-granite/granite-3.3-8b-instruct   $BIG train.objective=sft
train_eval pubmedqa_ministral-8b-p7_ce  student.source=mistralai/Ministral-8B-Instruct-2410  $BIG train.objective=sft