"""Step 3: evaluate a model on the test split (continuation only, never the prompt).

    python evaluate.py --config CFG --name teacher --model meta-llama/Llama-3.1-8B-Instruct
    python evaluate.py --config CFG --name d3lm_L3 --model runs/d3lm_L3/best --reference teacher
    python evaluate.py --config CFG --name floor --model copy_prompt --reference teacher     # no-skill baseline
    python evaluate.py --config CFG --name teacher --rescore --metrics-device cpu            # re-score saved answers
    python evaluate.py --config CFG --name d3lm_L3 --rescore --reference teacher --judge openai:gpt-4o-mini

metrics.json contains:
  acc / macro_f1 / em / f1   label or span vs gold
  gold_*   BLEU, ROUGE-1, ROUGE-L, chrF, cosine, BERTScore vs the gold text (PubMedQA: long_answer vs the
           justification after the label)
  fid_*    the same metrics vs the reference model's answer (--reference)
  *_floor_shuffled   the same metrics vs references of other questions (mean of --floor-shuffles shuffles)
  gold_ppl/bpc, fid_ppl/bpc   likelihood of the gold / reference answer given the prompt (needs the model)
  judge_<provider>   LLM judge 0-10 on a fixed subset
  failure  empty rate, stop rate, repetition, mean length
"""
import collections
import json
import os
import random
import re
import string

import numpy as np

from common import (EnergyMeter, collate, dataset_name, device_of, encode, generate, load_model, load_split,
                    load_tokenizer, parse_args, read_jsonl, write_json, write_jsonl)

TEXT_METRICS = ["bleu", "rouge1", "rougeL", "chrf", "cosine", "bertscore"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BERTSCORE_MODEL = "roberta-large"


# ---------------------------------------------------------------- task scores and references
def norm(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def task_scores(row, pred):
    if row["task"] == "extractive_qa":
        p = norm(pred.strip().split("\n")[0]).split()
        f1 = 0.0
        for g in row["gold"]:
            g = norm(g).split()
            same = sum((collections.Counter(p) & collections.Counter(g)).values())
            f1 = max(f1, 2 * same / (len(p) + len(g)) if same else 0.0)
        return {"em": float(any(p == norm(g).split() for g in row["gold"])), "f1": f1}
    if row["task"] == "yes_no_maybe":
        m = re.search(r"\b(yes|no|maybe)\b", pred.lower())
        return {"label": m and m.group(1), "acc": float(bool(m) and m.group(1) == row["gold"])}
    return {}


def reference_text(row):
    """Gold free text for the text metrics (None for span tasks)."""
    if row["task"] == "yes_no_maybe":
        return row.get("long_answer") or None
    if row["task"] in ("continuation", "generic") and row.get("gold"):
        return row["gold"] if isinstance(row["gold"], str) else row["gold"][0]
    return None


def answer_body(row, pred):
    """Drop the leading yes/no/maybe so only the justification is compared with the long answer."""
    if row["task"] == "yes_no_maybe":
        return re.sub(r"^\W*(yes|no|maybe)\b\W*", "", pred.strip(), flags=re.I)
    return pred


def ppl_reference(row):
    """Answer whose likelihood is measured: PubMedQA label + long answer, SQuAD span, or the gold text."""
    if row["task"] == "yes_no_maybe":
        return f"{row['gold']}. {row.get('long_answer', '')}".strip()
    if row["task"] == "extractive_qa":
        return row["gold"][0]
    return reference_text(row)


# ------------------------------------------------------------------------- text metrics
def text_metrics(cands, refs, which, device="cpu"):
    """Per-item scores for each metric in `which`. Empty texts score 0."""
    import sacrebleu
    from rouge_score import rouge_scorer
    out = {m: [0.0] * len(cands) for m in which}
    ok = [i for i in range(len(cands)) if cands[i].strip() and refs[i].strip()]
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    for i in ok:
        r = rouge.score(refs[i], cands[i])
        vals = {"bleu": sacrebleu.sentence_bleu(cands[i], [refs[i]]).score / 100,
                "chrf": sacrebleu.sentence_chrf(cands[i], [refs[i]]).score / 100,
                "rouge1": r["rouge1"].fmeasure, "rougeL": r["rougeL"].fmeasure}
        for m in which:
            if m in vals:
                out[m][i] = vals[m]
    if ok and "cosine" in which:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(EMBED_MODEL, device=device)
        a = enc.encode([cands[i] for i in ok], normalize_embeddings=True)
        b = enc.encode([refs[i] for i in ok], normalize_embeddings=True)
        for i, v in zip(ok, (a * b).sum(1)):
            out["cosine"][i] = float(v)
    if ok and "bertscore" in which:
        from bert_score import score
        f1 = score([cands[i] for i in ok], [refs[i] for i in ok], model_type=BERTSCORE_MODEL, device=device,
                   verbose=False)[2]
        for i, v in zip(ok, f1.tolist()):
            out["bertscore"][i] = v
    return out


def ci(values, n_boot=1000):
    v = np.asarray([x for x in values if x is not None], float)
    boots = np.random.default_rng(0).choice(v, (n_boot, len(v))).mean(1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "n": len(v)}


def add_text_scores(preds, idx, cands, refs, prefix, args, metrics):
    """Scores vs the right references, and the floor: the same answers vs other questions' references
    (refs rotated by a random non-zero offset, so no answer is paired with its own reference)."""
    scores = text_metrics(cands, refs, args.metrics, args.metrics_device)
    for m in args.metrics:
        for j, i in enumerate(idx):
            preds[i][f"{prefix}_{m}"] = scores[m][j]
        metrics[f"{prefix}_{m}"] = ci(scores[m])
    offsets = np.random.default_rng(0).integers(1, len(refs), args.floor_shuffles)
    floors = [text_metrics(cands, list(np.roll(refs, k)), args.metrics, args.metrics_device) for k in offsets]
    metrics[f"{prefix}_floor_shuffled"] = {m: float(np.mean([np.mean(f[m]) for f in floors])) for m in args.metrics}


# --------------------------------------------------------------------------- perplexity
def add_ppl_scores(model, tok, prompts, answers, prefix, cfg, metrics):
    """Perplexity (same-tokenizer comparisons) and bits per character (any tokenizer) of `answers`."""
    import torch
    import torch.nn.functional as F
    dev = next(model.parameters()).device
    nll = n_tok = n_char = 0.0
    with torch.no_grad():
        for i in range(0, len(prompts), 4):
            items = [encode(tok, p, a, cfg["max_prompt_tokens"], 1024) for p, a in zip(prompts[i:i + 4], answers[i:i + 4])]
            for e in items:
                e["loss_mask"][-1] = 0          # score the answer text, not the appended EOS
            b = {k: v.to(dev) for k, v in collate(items, tok.pad_token_id).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            mask = b["loss_mask"][:, 1:].float()
            loss = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), b["input_ids"][:, 1:], reduction="none")
            nll += (loss * mask).sum().item()
            n_tok += mask.sum().item()
            n_char += sum(len(a.strip()) for a in answers[i:i + 4])
    metrics[f"{prefix}_ppl"] = {"mean": float(np.exp(nll / n_tok))}
    metrics[f"{prefix}_bpc"] = {"mean": nll / np.log(2) / n_char}


# ---------------------------------------------------------------------------- LLM judge
JUDGE_PROMPT = """You are grading an answer to a question. You get the question with its context, a reference answer
written by an expert, and a candidate answer. Rate the candidate from 0 to 10 for correctness and for how well
its reasoning is supported by the context, using the reference as the gold standard:
10 = correct and fully supported, 7-9 = correct with minor gaps, 4-6 = partly correct or weakly supported,
1-3 = mostly wrong or unsupported, 0 = wrong, empty or unrelated. Ignore length and style.

Question and context:
{prompt}

Reference answer:
{reference}

Candidate answer:
{candidate}

Return only JSON: {{"score": <integer 0-10>}}"""


def call_judge(provider, prompt):
    """provider = 'openai:<model>' (OPENAI_API_KEY) or 'gemini:<model>' (GEMINI_API_KEY)."""
    kind, model = provider.split(":", 1)
    if kind == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(model=model, temperature=0, messages=[{"role": "user", "content": prompt}])
        text = r.choices[0].message.content
    else:
        from google import genai
        text = genai.Client().models.generate_content(model=model, contents=prompt).text
    m = re.search(r'"score"\s*:\s*(\d+)', text or "")
    return float(m.group(1)) if m else None


def run_judge(rows, preds, provider, n, cache_path, call=call_judge):
    """Judge the same n items (fixed seed) for every model; answers are cached so nothing is paid twice."""
    ids = sorted(r["id"] for r in rows)
    random.Random(0).shuffle(ids)
    subset = set(ids[:n])
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    by_id = {p["id"]: p["prediction"] for p in preds}
    for r in rows:
        if r["id"] in subset and r["id"] not in cache:
            ref = r["gold"] if isinstance(r["gold"], str) else ", ".join(r["gold"] or [])
            if r.get("long_answer"):
                ref = f"{ref}. {r['long_answer']}"
            cache[r["id"]] = call(provider, JUDGE_PROMPT.format(prompt=r["prompt"], reference=ref,
                                                                candidate=by_id[r["id"]]))
            json.dump(cache, open(cache_path, "w"))
    return {i: cache[i] for i in subset}


# --------------------------------------------------------------------------------- main
def main():
    args, cfg = parse_args(__doc__, lambda ap: (
        ap.add_argument("--name", required=True), ap.add_argument("--model"),
        ap.add_argument("--precision", default="bf16"), ap.add_argument("--tokenizer"),
        ap.add_argument("--reference", help="name of an evaluated model, usually the teacher"),
        ap.add_argument("--rescore", action="store_true", help="re-score saved answers, no generation"),
        ap.add_argument("--metrics", nargs="*", default=TEXT_METRICS),
        ap.add_argument("--metrics-device", default="cuda"),
        ap.add_argument("--floor-shuffles", type=int, default=5),
        ap.add_argument("--judge", nargs="*", default=[], help="e.g. openai:gpt-4o-mini"),
        ap.add_argument("--judge-n", type=int, default=100),
        ap.add_argument("--no-ppl", action="store_true"),
        ap.add_argument("--chat-template", action="store_true", help="wrap prompts in the model's chat template"),
        ap.add_argument("--out", default="eval")))
    import torch
    if not torch.cuda.is_available():
        args.metrics_device = "cpu"
    model = tok = None
    if args.model and args.model != "copy_prompt" and not (args.rescore and args.no_ppl):
        tok = load_tokenizer(args.tokenizer or args.model)
        model = load_model(args.model, args.precision, device_of(cfg))
    meter = EnergyMeter(cfg["energy"]["enabled"] and not args.rescore)

    for spec in cfg["datasets"]:
        ds = dataset_name(spec)
        out_dir = os.path.join(args.out, args.name, ds)
        rows = load_split(cfg, spec, "test")
        if args.rescore:
            saved = {p["id"]: p for p in read_jsonl(f"{out_dir}/predictions.jsonl")}
            outs = [{"text": saved[r["id"]]["prediction"], "n_new_tokens": saved[r["id"]]["n_new_tokens"],
                     "stopped": saved[r["id"]]["stopped"]} for r in rows]
            energy = json.load(open(f"{out_dir}/metrics.json")).get("generation_energy")
        else:
            meter.start()
            if model is None:   # copy_prompt baseline
                outs = [{"text": " ".join(r["prompt"].split()[-32:]), "n_new_tokens": 0, "stopped": True} for r in rows]
            else:
                outs = generate(model, tok, [r["prompt"] for r in rows], cfg["max_new_tokens"], 16,
                                cfg["max_prompt_tokens"], chat=args.chat_template)
            energy = meter.stop()

        preds = [{"id": r["id"], "prediction": o["text"], "n_new_tokens": o["n_new_tokens"], "stopped": o["stopped"],
                  **task_scores(r, o["text"])} for r, o in zip(rows, outs)]
        metrics = {"model": args.model, "precision": args.precision, "n": len(preds), "reference": args.reference,
                   "max_new_tokens": cfg["max_new_tokens"], "generation_energy": energy}
        for k in ("em", "f1", "acc"):
            if k in preds[0]:
                metrics[k] = ci([p[k] for p in preds])
        if rows[0]["task"] == "yes_no_maybe":
            f1s = []
            for lab in ("yes", "no", "maybe"):
                tp = sum(p["label"] == lab and r["gold"] == lab for p, r in zip(preds, rows))
                wrong = sum((p["label"] == lab) != (r["gold"] == lab) for p, r in zip(preds, rows))
                f1s.append(2 * tp / (2 * tp + wrong) if tp else 0.0)
            metrics["macro_f1"] = float(np.mean(f1s))

        idx = [i for i, r in enumerate(rows) if reference_text(r)]
        if idx:
            add_text_scores(preds, idx, [answer_body(rows[i], preds[i]["prediction"]) for i in idx],
                            [reference_text(rows[i]) for i in idx], "gold", args, metrics)
        if args.reference:
            refs = {p["id"]: p["prediction"] for p in read_jsonl(f"{args.out}/{args.reference}/{ds}/predictions.jsonl")}
            add_text_scores(preds, list(range(len(rows))), [p["prediction"] for p in preds],
                            [refs[r["id"]] for r in rows], "fid", args, metrics)
        if model is not None and not args.no_ppl:
            gold_rows = [r for r in rows if ppl_reference(r)]
            add_ppl_scores(model, tok, [r["prompt"] for r in gold_rows], [ppl_reference(r) for r in gold_rows],
                           "gold", cfg, metrics)
            if args.reference:
                add_ppl_scores(model, tok, [r["prompt"] for r in rows], [refs[r["id"]] for r in rows], "fid", cfg, metrics)
        for provider in args.judge:
            os.makedirs(out_dir, exist_ok=True)
            scores = run_judge(rows, preds, provider, args.judge_n, f"{out_dir}/judge_{provider.replace(':', '_')}.json")
            metrics[f"judge_{provider}"] = ci(list(scores.values()))

        texts = [p["prediction"] for p in preds]
        repeated = []   # share of repeated 4-grams per answer
        for t in texts:
            w = t.split()
            grams = list(zip(w, w[1:], w[2:], w[3:]))
            repeated.append(1 - len(set(grams)) / len(grams) if grams else 0.0)
        metrics["failure"] = {"empty_rate": float(np.mean([not t.strip() for t in texts])),
                              "stop_rate": float(np.mean([p["stopped"] for p in preds])),
                              "degenerate_rate": float(np.mean([r > 0.5 for r in repeated])),
                              "mean_new_tokens": float(np.mean([p["n_new_tokens"] for p in preds]))}
        write_jsonl(preds, f"{out_dir}/predictions.jsonl")
        write_json(metrics, f"{out_dir}/metrics.json")
        shown = [k for k in ("acc", "macro_f1", "f1", "gold_rougeL", "gold_bertscore", "fid_rougeL", "gold_ppl", "fid_ppl")
                 if k in metrics]
        print(f"{args.name} / {ds}: n={len(preds)}, " + ", ".join(
            f"{k}={metrics[k]['mean'] if isinstance(metrics[k], dict) else metrics[k]:.3f}" for k in shown))


if __name__ == "__main__":
    main()