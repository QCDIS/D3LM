#!/bin/bash
# Energy per query for every model architecture on every dataset. Run with NOTHING else on the GPU (NVML measures
# the whole GPU).
#   DRY=1 bash run_energy.sh                      # print the commands only
#   bash run_energy.sh 2>&1 | tee -a logs/run_energy.log
#   python energy_table.py                        # then: energy per query, one-time energy, break-even per run
#
# Every request generates exactly max_new_tokens of the dataset's config (EOS ignored), batch 32, so the energy per
# query depends only on the architecture and the dataset (prompt length, answer length), not on the trained weights.
# One untrained checkpoint per pruned depth (7, 14, 5 layers) therefore stands in for every student of that depth.
# Low-rank healing layers (0.5 M weights per seam) are not benchmarked separately: their cost is negligible.
# Output: bench/<dataset>__<architecture>.json. About 3-5 min per model, ~2 h in total. Finished files are skipped.
set -f
mkdir -p logs bench
run() { echo "+ $*"; [ -n "$DRY" ] || "$@"; }
B="--batch-sizes 32 --n-prompts 64"

arch_ckpt() {   # arch_ckpt <config> <name> <layers>: untrained pruned teacher (1 step, lr 0), only its shape matters
  [ -d runs/$2/best ] || run python train.py --config $1 --set run_name=$2 student.num_layers=$3 \
      train.objective=sft train.max_steps=1 train.lr=0 train.eval_every=1
}
arch_ckpt configs/climate_llama_L7.yaml bench_llama31-8b-p7 7
arch_ckpt configs/climate_llama_L7.yaml bench_llama31-8b-p14 14
arch_ckpt configs/climate_llama_L7.yaml bench_llama31-8b-p5 5
arch_ckpt configs/code_qwen_L7.yaml bench_qwen25coder-7b-p7 7

bench() {   # bench <config> <dataset> <architecture> <model> [extra args]
  local cfg=$1 ds=$2 a=$3 m=$4; shift 4
  [ -f bench/${ds}__$a.json ] || run python energy_benchmark.py --config $cfg --name ${ds}__$a --model $m $B "$@" \
      2>&1 | tee logs/bench_${ds}__$a.log
}
SMALL="qwen25-1.5b=Qwen/Qwen2.5-1.5B-Instruct smollm2-1.7b=HuggingFaceTB/SmolLM2-1.7B-Instruct gemma3-1b=google/gemma-3-1b-it"

for spec in "configs/pubmedqa_full_llama_L7_chat.yaml pubmedqa" "configs/climate_llama_L7.yaml climate"; do
  set -- $spec; cfg=$1; ds=$2
  T=meta-llama/Llama-3.1-8B-Instruct
  bench $cfg $ds teacher      $T
  bench $cfg $ds teacher_int8 $T --precision int8
  bench $cfg $ds teacher_nf4  $T --precision nf4
  bench $cfg $ds p14 runs/bench_llama31-8b-p14/best
  bench $cfg $ds p7  runs/bench_llama31-8b-p7/best
  [ $ds = climate ] && bench $cfg $ds p5 runs/bench_llama31-8b-p5/best
  bench $cfg $ds llama32-1b meta-llama/Llama-3.2-1B-Instruct
  for s in $SMALL; do bench $cfg $ds ${s%%=*} ${s#*=}; done
done

cfg=configs/code_qwen_L7.yaml; T=Qwen/Qwen2.5-Coder-7B-Instruct
bench $cfg code teacher      $T
bench $cfg code teacher_int8 $T --precision int8
bench $cfg code teacher_nf4  $T --precision nf4
bench $cfg code p7 runs/bench_qwen25coder-7b-p7/best
bench $cfg code qwen25coder-1.5b Qwen/Qwen2.5-Coder-1.5B-Instruct
for s in $SMALL; do bench $cfg code ${s%%=*} ${s#*=}; done
echo "all done"