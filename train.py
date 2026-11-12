"""Step 2: distil a student from the teacher (or train the same student with SFT).

    python train.py --config configs/pubmedqa_llama.yaml                               # D3LM, 3 layers
    python train.py --config configs/pubmedqa_llama.yaml --set train.objective=sft     # fair SFT baseline
    python train.py --config configs/pubmedqa_llama.yaml --layers 1 2 3 4 6            # depth sweep

KD and SFT use exactly the same loop, data, optimiser, schedule and frozen parts; SFT only
drops the KD term (and never runs the teacher). Output per run in runs/<run_name>/:
best/ (checkpoint with the lowest validation loss), log.jsonl, summary.json (incl. energy).
"""
import copy
import gc
import math
import os
import time

import torch
import torch.nn.functional as F

from common import (EnergyMeter, build_student, collate, device_of, encode, load_model, load_tokenizer,
                    parse_args, set_seed, teacher_tag, training_rows, write_json)


# ================================================================================ loss
def kd_loss(s_logits, t_logits, input_ids, loss_mask, ce_weight, temperature, divergence="fkl", chunk=2048,
            skew_alpha=0.1):
    """(1 - w) * T^2 * D + w * CE over the completion tokens only (next-token shift).

    D (p = teacher, q = student, both at temperature T):
      fkl   KL(p || q)                              forward KL (original D3LM)
      rkl   KL(q || p)                              reverse KL
      skl   KL(p || a*p + (1-a)*q)                  skew KL         (DistiLLM, Ko et al., ICML 2024)
      srkl  KL(q || (1-a)*p + a*q)                  skew reverse KL (DistiLLM)
    a = skew_alpha in (0, 1); a -> 0 recovers fkl / rkl. Mixing a little of p into the
    student side keeps the divergence bounded when the student is far from the teacher.

    Only target positions are kept before the softmax, and the KL is computed in chunks,
    so large vocabularies (128k-152k) fit in memory. t_logits=None means SFT (CE only).
    """
    if divergence in ("skl", "srkl") and not 0.0 < skew_alpha < 1.0:
        raise ValueError("skew_alpha must be in (0, 1)")
    mask = loss_mask[:, 1:].bool()
    targets = input_ids[:, 1:][mask]
    s = s_logits[:, :-1][mask].float()
    vocab = s.shape[-1] if t_logits is None else min(s.shape[-1], t_logits.shape[-1])
    s = s[:, :vocab]
    ce = F.cross_entropy(s, targets)
    if t_logits is None:
        return ce, torch.zeros(()), ce.detach()
    t = t_logits[:, :-1][mask][:, :vocab].float()
    kl = 0.0
    for i in range(0, s.shape[0], chunk):
        ls = F.log_softmax(s[i:i + chunk] / temperature, -1)
        lt = F.log_softmax(t[i:i + chunk] / temperature, -1)
        if divergence == "fkl":
            d = lt.exp() * (lt - ls)
        elif divergence == "rkl":
            d = ls.exp() * (ls - lt)
        elif divergence == "skl":    # log of the mixture a*p + (1-a)*q, computed stably in log space
            log_m = torch.logaddexp(lt + math.log(skew_alpha), ls + math.log1p(-skew_alpha))
            d = lt.exp() * (lt - log_m)
        elif divergence == "srkl":
            log_m = torch.logaddexp(lt + math.log1p(-skew_alpha), ls + math.log(skew_alpha))
            d = ls.exp() * (ls - log_m)
        else:
            raise ValueError(f"unknown divergence {divergence!r}")
        kl = kl + d.sum()
    kl = kl / max(s.shape[0], 1) * temperature ** 2
    return (1 - ce_weight) * kl + ce_weight * ce, kl.detach(), ce.detach()


# ============================================================================ training
def forward(model, batch, bf16):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and torch.cuda.is_available()):
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits


@torch.no_grad()
def validate(student, teacher, val, cfg, pad_id, dev):
    """Validation CE on the target completions (comparable between KD and SFT) and KL to the teacher."""
    student.eval()
    t = cfg["train"]
    ce_sum = kl_sum = n_tok = 0.0
    for i in range(0, len(val), t["batch_size"]):
        b = {k: v.to(dev) for k, v in collate(val[i:i + t["batch_size"]], pad_id).items()}
        s = forward(student, b, t["bf16"])
        n = int(b["loss_mask"][:, 1:].sum())
        tl = forward(teacher, b, t["bf16"]) if teacher is not None else None
        _, kl, ce = kd_loss(s, tl, b["input_ids"], b["loss_mask"], 0.0, t["temperature"], t["divergence"],
                            skew_alpha=t["skew_alpha"])
        ce_sum, kl_sum, n_tok = ce_sum + ce.item() * n, kl_sum + float(kl) * n, n_tok + n
    student.train()
    out = {"val_ce": ce_sum / n_tok, "val_ppl": math.exp(min(ce_sum / n_tok, 50))}
    if teacher is not None:
        out["val_kl"] = kl_sum / n_tok
    return out


def train_one(cfg, teacher, build_teacher, teacher_tok, dev):
    t = cfg["train"]
    set_seed(cfg["seed"])
    name = cfg["run_name"] or (f"{teacher_tag(cfg)}_{'+'.join(cfg['datasets'])}_{cfg['student']['init']}"
                               f"-L{cfg['student']['num_layers']}_{t['objective']}_s{cfg['seed']}")
    run_dir = os.path.join(cfg["output_dir"], name)
    os.makedirs(run_dir, exist_ok=True)
    write_json(cfg, os.path.join(run_dir, "config.json"))

    student, tok, info = build_student(cfg, build_teacher, teacher_tok)
    student.to(dev).train()
    if t.get("gradient_checkpointing"):   # recompute activations in the backward pass: less memory, ~30% slower
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        student.config.use_cache = False
    print(f"[{name}] student: {info}")
    pad = tok.pad_token_id
    enc = lambda rows: [encode(tok, r["prompt"], r["completion"], cfg["max_prompt_tokens"], cfg["max_new_tokens"])
                        for r in rows]
    train, val = enc(training_rows(cfg, "train")), enc(training_rows(cfg, "validation"))

    params = [p for p in student.parameters() if p.requires_grad]
    if t.get("optimizer", "adamw") == "adamw_8bit":   # 8-bit optimizer state: ~2 bytes/param instead of 8
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=float(t["lr"]))
    else:
        opt = torch.optim.AdamW(params, lr=float(t["lr"]))
    warm = int(t["warmup_ratio"] * t["max_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, t["max_steps"] - warm))))
    use_teacher = t["objective"] == "kd"
    meter = EnergyMeter(cfg["energy"]["enabled"])
    idle = meter.measure_idle(cfg["energy"]["idle_seconds"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    order, pos = torch.randperm(len(train), generator=gen).tolist(), 0
    best, best_step, bad, history, step = float("inf"), 0, 0, [], 0
    log = open(os.path.join(run_dir, "log.jsonl"), "w")
    t0 = time.time()
    meter.start()
    while step < t["max_steps"]:
        for _ in range(t["grad_accum"]):
            if pos + t["batch_size"] > len(order):
                order, pos = torch.randperm(len(train), generator=gen).tolist(), 0
            b = {k: v.to(dev) for k, v in collate([train[i] for i in order[pos:pos + t["batch_size"]]], pad).items()}
            pos += t["batch_size"]
            s = forward(student, b, t["bf16"])
            with torch.no_grad():
                tl = forward(teacher, b, t["bf16"]) if use_teacher else None
            loss, kl, ce = kd_loss(s, tl, b["input_ids"], b["loss_mask"], t["ce_weight"] if use_teacher else 1.0,
                                   t["temperature"], t["divergence"], skew_alpha=t["skew_alpha"])
            (loss / t["grad_accum"]).backward()
        torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == 1:
            log.write(f'{{"step": {step}, "loss": {loss.item():.4f}, "kl": {float(kl):.4f}, "ce": {ce.item():.4f}, '
                      f'"lr": {sched.get_last_lr()[0]:.2e}, "elapsed_s": {time.time() - t0:.0f}}}\n')
            log.flush()
        if step % t["eval_every"] == 0 or step == t["max_steps"]:
            ev = {"step": step, **validate(student, teacher if use_teacher else None, val, cfg, pad, dev)}
            history.append(ev)
            print(f"[{name}] {ev}")
            if ev["val_ce"] < best:
                best, best_step, bad = ev["val_ce"], step, 0
                student.save_pretrained(os.path.join(run_dir, "best"))
                tok.save_pretrained(os.path.join(run_dir, "best"))
            else:
                bad += 1
                if t["patience"] and bad >= t["patience"]:
                    print(f"[{name}] early stop at step {step}")
                    break
    energy = meter.stop()
    log.close()
    summary = {"run": name, "steps": step, "best_step": best_step, "best_val_ce": best, "student": info,
               "teacher_used": use_teacher, "n_train": len(train), "n_val": len(val),
               "wall_time_s": time.time() - t0, "energy": energy, "idle_w": idle,
               "gpu": meter.gpu_info() if meter.mode != "none" else None, "history": history}
    write_json(summary, os.path.join(run_dir, "summary.json"))
    print(f"[{name}] done: best val_ce {best:.4f} at step {best_step}, energy {energy.get('kwh')} kWh")
    del student, opt
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return run_dir


def main():
    args, cfg = parse_args(__doc__, lambda ap: ap.add_argument(
        "--layers", type=int, nargs="*", help="train one student per depth (depth sweep)"))
    dev = device_of(cfg)
    teacher_tok = load_tokenizer(cfg["teacher"])
    kd = cfg["train"]["objective"] == "kd"
    plain = cfg["teacher_precision"] in ("bf16", "fp16", "fp32")
    needs_weights = cfg["student"]["init"] != "pretrained"
    # KD teacher on the GPU. Students are built from float weights, so an int8/nf4 teacher (or SFT,
    # which never runs the teacher) uses a temporary bf16 copy on the CPU for building only.
    teacher = load_model(cfg["teacher"], cfg["teacher_precision"], dev) if kd else None
    # the student is built from `student.source` (another model to prune) or, by default, the teacher
    source = cfg["student"].get("source") or cfg["teacher"]
    if source != cfg["teacher"]:
        teacher_tok = load_tokenizer(source)
        if kd and teacher_tok.get_vocab() != load_tokenizer(cfg["teacher"]).get_vocab():
            raise ValueError("student.source and teacher tokenizers differ: use train.objective=sft")
    build_teacher = None
    if needs_weights:
        build_teacher = teacher if (kd and plain and source == cfg["teacher"]) else load_model(source, "bf16", "cpu")
    if teacher is not None:
        teacher.requires_grad_(False)
    for L in (args.layers or [cfg["student"]["num_layers"]]):
        c = copy.deepcopy(cfg)
        c["student"]["num_layers"] = L
        if args.layers and cfg["run_name"]:
            c["run_name"] = f"{cfg['run_name']}_L{L}"
        train_one(c, teacher, build_teacher, teacher_tok, dev)


if __name__ == "__main__":
    main()