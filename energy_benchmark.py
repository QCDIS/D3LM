"""Step 4: inference latency / throughput / energy per request. Run with NOTHING else on the GPU.

HF generate, several batch sizes (replaces the legacy 6-prompt, batch-1 measurement):
    python energy_benchmark.py --config configs/pubmedqa_llama.yaml --name teacher --model meta-llama/Llama-3.1-8B-Instruct
    python energy_benchmark.py --config configs/pubmedqa_llama.yaml --name d3lm_L3 --model runs/<run>/best

vLLM serving under load (Poisson arrivals at several request rates, plus offline max throughput):
    python energy_benchmark.py --config ... --name teacher --model meta-llama/Llama-3.1-8B-Instruct --vllm
    python energy_benchmark.py --config ... --name teacher_awq --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --vllm --quantization awq

Every request generates exactly max_new_tokens (EOS ignored), so all models do the same work.
Each measurement window lasts at least --min-seconds and is repeated --repeats times; idle power
is measured first so results include total and idle-subtracted energy. Output: bench/<name>.json
"""
import asyncio
import time
import uuid

import numpy as np
import torch

from common import EnergyMeter, device_of, generate, load_model, load_split, load_tokenizer, parse_args, write_json


def bench_hf(args, cfg, prompts, meter):
    tok = load_tokenizer(args.model)
    model = load_model(args.model, args.precision, device_of(cfg))
    results = []
    for bs in args.batch_sizes:
        generate(model, tok, prompts[:2 * bs], cfg["max_new_tokens"], bs, cfg["max_prompt_tokens"], ignore_eos=True)  # warm-up
        reps = []
        for _ in range(args.repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            n_req = n_tok = 0
            meter.start()
            t0 = time.perf_counter()
            while n_req == 0 or time.perf_counter() - t0 < args.min_seconds:
                outs = generate(model, tok, prompts, cfg["max_new_tokens"], bs, cfg["max_prompt_tokens"], ignore_eos=True)
                n_req, n_tok = n_req + len(outs), n_tok + sum(o["n_new_tokens"] for o in outs)
            e = meter.stop()
            rec = {"requests": n_req, "tokens": n_tok, "seconds": e["seconds"], "requests_per_s": n_req / e["seconds"],
                   "tokens_per_s": n_tok / e["seconds"], "energy": e,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None}
            if "energy_j" in e:
                rec.update({"j_per_request": e["energy_j"] / n_req, "j_per_token": e["energy_j"] / n_tok})
            if "dynamic_energy_j" in e:
                rec["dyn_j_per_request"] = e["dynamic_energy_j"] / n_req
            reps.append(rec)
        summary = {k: float(np.mean([r[k] for r in reps])) for k in reps[0] if isinstance(reps[0][k], (int, float))}
        results.append({"batch_size": bs, "mean": summary, "repeats": reps})
        print(f"batch {bs}: {summary.get('tokens_per_s', 0):.0f} tok/s, {summary.get('j_per_request')} J/request")
    return {"hf": results}


def bench_vllm(args, cfg, prompts, meter):
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    async def one(engine, prompt, sp, delay, t0):
        await asyncio.sleep(max(0.0, t0 + delay - time.perf_counter()))
        start, first, n = time.perf_counter(), None, 0
        async for out in engine.generate(prompt, sp, request_id=str(uuid.uuid4())):
            n = len(out.outputs[0].token_ids)
            if first is None and n:
                first = time.perf_counter()
        end = time.perf_counter()
        return {"ttft": first - start, "tpot": (end - first) / max(n - 1, 1), "e2e": end - start, "tokens": n}

    def summarise(recs, e):
        p = lambda k, q: float(np.percentile([r[k] for r in recs], q))
        out = {"requests": len(recs), "seconds": e["seconds"], "requests_per_s": len(recs) / e["seconds"],
               "tokens_per_s": sum(r["tokens"] for r in recs) / e["seconds"],
               "ttft_p50": p("ttft", 50), "ttft_p99": p("ttft", 99), "tpot_p50": p("tpot", 50),
               "tpot_p99": p("tpot", 99), "e2e_p50": p("e2e", 50), "e2e_p99": p("e2e", 99),
               "slo_ok": float(np.mean([r["ttft"] <= args.ttft_slo and r["tpot"] <= args.tpot_slo for r in recs])),
               "energy": e}
        if "energy_j" in e:
            out["j_per_request"] = e["energy_j"] / len(recs)
        if "dynamic_energy_j" in e:
            out["dyn_j_per_request"] = e["dynamic_energy_j"] / len(recs)
        return out

    async def run():
        engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=args.model, quantization=args.quantization, dtype="bfloat16", gpu_memory_utilization=0.85))
        sp = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"], ignore_eos=True)
        n = args.n_requests
        await asyncio.gather(*[one(engine, prompts[i], sp, 0, time.perf_counter()) for i in range(8)])   # warm-up
        res = {"offline": None, "online": []}
        meter.start()
        t0 = time.perf_counter()
        res["offline"] = summarise(await asyncio.gather(
            *[one(engine, prompts[i % len(prompts)], sp, 0, t0) for i in range(n)]), meter.stop())
        for rate in args.rates:
            arrivals = np.cumsum(np.random.default_rng(cfg["seed"]).exponential(1 / rate, n))
            meter.start()
            t0 = time.perf_counter()
            recs = await asyncio.gather(*[one(engine, prompts[i % len(prompts)], sp, arrivals[i], t0) for i in range(n)])
            s = summarise(recs, meter.stop())
            s["offered_rps"] = rate
            res["online"].append(s)
            print(f"rate {rate}/s: p99 TTFT {s['ttft_p99']:.2f}s, SLO met {s['slo_ok']:.0%}, {s.get('j_per_request')} J/request")
        ok = [s["offered_rps"] for s in res["online"] if s["slo_ok"] >= 0.9 and s["requests_per_s"] >= 0.95 * s["offered_rps"]]
        res["max_rate_under_slo"] = max(ok) if ok else None
        return res

    return asyncio.run(run())


def main():
    args, cfg = parse_args(__doc__, lambda ap: (
        ap.add_argument("--name", required=True), ap.add_argument("--model", required=True),
        ap.add_argument("--precision", default="bf16"), ap.add_argument("--n-prompts", type=int, default=256),
        ap.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 8, 32]),
        ap.add_argument("--repeats", type=int, default=3), ap.add_argument("--min-seconds", type=float, default=30),
        ap.add_argument("--vllm", action="store_true"), ap.add_argument("--quantization", default=None),
        ap.add_argument("--rates", type=float, nargs="*", default=[1, 2, 4, 8, 16]),
        ap.add_argument("--n-requests", type=int, default=500),
        ap.add_argument("--ttft-slo", type=float, default=1.0), ap.add_argument("--tpot-slo", type=float, default=0.1),
        ap.add_argument("--out", default=None)))
    prompts = [r["prompt"] for r in load_split(cfg, cfg["datasets"][0], "test")][:args.n_prompts]
    meter = EnergyMeter(cfg["energy"]["enabled"])
    idle = meter.measure_idle(cfg["energy"]["idle_seconds"])
    res = bench_vllm(args, cfg, prompts, meter) if args.vllm else bench_hf(args, cfg, prompts, meter)
    res.update({"name": args.name, "model": args.model, "precision": args.precision, "idle_w": idle,
                "max_new_tokens": cfg["max_new_tokens"], "gpu": meter.gpu_info() if meter.mode != "none" else None})
    write_json(res, args.out or f"bench/{args.name}.json")


if __name__ == "__main__":
    main()
