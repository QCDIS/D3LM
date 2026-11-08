"""Step 1: create the train/validation/test splits and the teacher's training targets.

    python prepare_data.py --config configs/pubmedqa_llama.yaml
    python prepare_data.py --config configs/pubmedqa_llama.yaml --vllm        # faster teacher generation

Writes data/splits/<dataset>/{train,validation,test}.jsonl (+ manifest with sha256) and
data/teacher/<teacher>/<dataset>_{train,validation}.jsonl. The energy used by the teacher is
saved in data/teacher/<teacher>/manifest.json (data-generation cost for break_even.py).
"""
import os

from common import (EnergyMeter, device_of, generate, load_model, load_split, load_tokenizer, make_splits,
                    parse_args, read_json, targets_path, teacher_tag, write_json, write_jsonl)


def vllm_generate(model_name, prompts, max_new_tokens, chat=False):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    if chat:
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp)
    else:
        outs = llm.generate(prompts, sp)
    return [{"text": o.outputs[0].text, "n_new_tokens": len(o.outputs[0].token_ids),
             "stopped": o.outputs[0].finish_reason == "stop"} for o in outs]


def main():
    args, cfg = parse_args(__doc__, lambda ap: (
        ap.add_argument("--vllm", action="store_true", help="generate teacher targets with vLLM"),
        ap.add_argument("--splits-only", action="store_true")))
    for spec in cfg["datasets"]:
        make_splits(cfg, spec)
    if args.splits_only:
        return

    meter = EnergyMeter(cfg["energy"]["enabled"])
    model = tok = None
    if not args.vllm:
        tok = load_tokenizer(cfg["teacher"])
        model = load_model(cfg["teacher"], cfg["teacher_precision"], device_of(cfg))
    manifest_path = os.path.join(cfg["targets_dir"], teacher_tag(cfg), "manifest.json")
    manifest = read_json(manifest_path) if os.path.exists(manifest_path) else {}
    meter.measure_idle(cfg["energy"]["idle_seconds"])
    for spec in cfg["datasets"]:
        for split in ("train", "validation"):
            out_path = targets_path(cfg, spec, split)
            if os.path.exists(out_path):
                print(f"{out_path} exists, skipping")
                continue
            rows = load_split(cfg, spec, split)
            prompts = [r["prompt"] for r in rows]
            meter.start()
            chat = cfg.get("teacher_chat_template", False)
            outs = vllm_generate(cfg["teacher"], prompts, cfg["max_new_tokens"], chat) if args.vllm else \
                generate(model, tok, prompts, cfg["max_new_tokens"], 16, cfg["max_prompt_tokens"], chat=chat)
            energy = meter.stop()
            write_jsonl([{"id": r["id"], "completion": o["text"], "stopped": o["stopped"]}
                         for r, o in zip(rows, outs)], out_path)
            manifest[os.path.basename(out_path)[:-6]] = {"n": len(rows), "energy": energy,
                                                         "new_tokens": sum(o["n_new_tokens"] for o in outs)}
            print(f"{out_path}: {len(rows)} completions, {energy.get('kwh')} kWh")
            write_json(manifest, manifest_path)


if __name__ == "__main__":
    main()