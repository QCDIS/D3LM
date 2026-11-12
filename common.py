"""Shared helpers: config, datasets and splits, model loading, students, tokenisation,
generation and GPU energy measurement. Used by every script in this folder."""
import copy
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
import yaml

# =============================================================================== config
DEFAULTS = {
    "teacher": "meta-llama/Llama-3.1-8B-Instruct",
    "teacher_precision": "bf16",          # bf16 | fp16 | fp32 | int8 | nf4
    "teacher_tag": None,                  # folder name for teacher outputs (default: model name)
    "teacher_chat_template": False,       # generate teacher targets with the teacher's chat template
    "datasets": ["pubmedqa"],             # wikitext2 | squad | pubmedqa | jsonl:/path/file.jsonl
    "splits_dir": "data/splits",
    "targets_dir": "data/teacher",
    "train_size": 20000, "val_size": 500, "test_size": 1000,
    "max_prompt_tokens": 512,
    "max_new_tokens": 128,                # teacher targets, evaluation and benchmarks
    "output_dir": "runs",
    "run_name": None,
    "seed": 1,
    "device": "auto",
    "student": {
        "init": "random",                 # random (D3LM) | teacher_layers | pretrained
        "num_layers": 3,
        "layer_ids": None,                # teacher_layers: which teacher layers to keep (default: uniform)
        "pretrained": None,               # pretrained: HF id, e.g. meta-llama/Llama-3.2-1B-Instruct
        "source": None,                   # teacher_layers / random: model to take layers from (default: the teacher)
        "freeze": ["embeddings", "lm_head"],
    },
    "train": {
        "objective": "kd",                # kd | sft   (same loop; sft drops the KD term)
        "targets": "teacher",             # teacher | gold   (what the student learns to produce)
        "ce_weight": 0.2,                 # loss = (1 - ce_weight) * KD + ce_weight * CE
        "temperature": 2.0,
        "divergence": "fkl",              # fkl | rkl | skl | srkl
        "skew_alpha": 0.1,                # skl / srkl only (DistiLLM default)
        "lr": 1.0e-4,
        "batch_size": 4,
        "grad_accum": 8,
        "max_steps": 1500,
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "eval_every": 100,
        "patience": 5,                    # early stopping (evaluations without improvement); 0 = off
        "bf16": True,
        "optimizer": "adamw",             # adamw | adamw_8bit (bitsandbytes; for wide students, e.g. 10 layers)
        "gradient_checkpointing": False,  # less activation memory, ~30% slower
        "train_limit": None,              # use only N training examples (data-size sweeps)
        "drop_truncated": False,          # skip teacher answers that hit max_new_tokens
    },
    "energy": {"enabled": True, "idle_seconds": 20},
}


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(_read_yaml(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def load_config(path=None, overrides=()):
    """DEFAULTS <- YAML file (may name a `base:` file) <- `key=value` overrides (dotted keys)."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = _merge(cfg, _read_yaml(path))
    for item in overrides or ():
        key, value = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    return cfg


def teacher_tag(cfg):
    return cfg.get("teacher_tag") or cfg["teacher"].rstrip("/").split("/")[-1]


def device_of(cfg):
    return ("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def text_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:16]


# ============================================================================= datasets
# Every dataset becomes {"id", "dataset", "task", "prompt", "gold", "group"}.
# QA prompts always contain the question (the legacy evaluation cut prompts to 30% of their
# words, which removed the SQuAD question).

def _hf(path, name=None, split=None):
    from datasets import load_dataset
    return load_dataset(path, name, split=split)


def _rank(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _wikitext2(split, prompt_words=64, gold_words=64):
    fixes = [(" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" ,", ","), (" .", "."), (" 's", "'s")]
    rows, title = [], "untitled"
    for i, r in enumerate(_hf("Salesforce/wikitext", "wikitext-2-raw-v1", split)):
        line = r["text"].strip()
        if line.startswith("= ") and not line.startswith("= ="):
            title = line.strip("= ")
            continue
        if not line or line.startswith("="):
            continue
        for a, b in fixes:
            line = line.replace(a, b)
        w = line.split()
        if len(w) >= prompt_words + gold_words:
            rows.append({"id": f"wikitext2-{split}-{i}", "dataset": "wikitext2", "task": "continuation",
                         "prompt": " ".join(w[:prompt_words]),
                         "gold": " ".join(w[prompt_words:prompt_words + gold_words]), "group": title})
    return rows


def _squad(split):
    # official dev set = our test; validation = 10% of training articles (whole articles)
    rows = []
    for r in _hf("rajpurkar/squad", split="validation" if split == "test" else "train"):
        in_val = _rank(r["title"]) % 10 == 0
        if split != "test" and in_val != (split == "validation"):
            continue
        rows.append({"id": f"squad-{r['id']}", "dataset": "squad", "task": "extractive_qa",
                     "prompt": ("Answer the question with a short span copied from the context.\n\n"
                                f"Context: {r['context']}\nQuestion: {r['question']}\nAnswer:"),
                     "gold": list(dict.fromkeys(r["answers"]["text"])), "group": r["title"]})
    return rows


def _pubmedqa(split):
    # test = PQA-L (1,000 expert-labelled); train/validation = PQA-A (artificial)
    ds = _hf("qiaojin/PubMedQA", "pqa_labeled" if split == "test" else "pqa_artificial", "train")
    rows = []
    for r in ds:
        pid = str(r["pubid"])
        if split != "test" and (_rank(pid) % 50 == 0) != (split == "validation"):
            continue
        rows.append({"id": f"pubmedqa-{pid}", "dataset": "pubmedqa", "task": "yes_no_maybe",
                     "prompt": ("Answer the biomedical research question based on the abstract. Start with "
                                "yes, no, or maybe, then give a one-sentence justification.\n\n"
                                f"Abstract: {' '.join(r['context']['contexts'])}\n"
                                f"Question: {r['question']}\nAnswer:"),
                     "gold": r["final_decision"], "long_answer": r.get("long_answer", ""), "group": pid})
    return rows


def _jsonl(path, split):
    """Own dataset: lines {"prompt", "gold"?, "task"?, "split"?, "group"?}."""
    name = os.path.basename(path).split(".")[0]
    rows = []
    for i, r in enumerate(read_jsonl(path)):
        group = str(r.get("group", i))
        s = r.get("split") or ("test" if _rank(group) % 10 == 0 else "validation" if _rank(group) % 10 == 1 else "train")
        if s == split:
            rows.append({"id": f"{name}-{i}", "dataset": name, "task": r.get("task", "generic"),
                         "prompt": r["prompt"], "gold": r.get("gold"), "group": group})
    return rows


def dataset_name(spec):
    return os.path.basename(spec[6:]).split(".")[0] if spec.startswith("jsonl:") else spec


def dataset_pool(spec, split):
    if spec.startswith("jsonl:"):
        return _jsonl(spec[6:], split)
    return {"wikitext2": _wikitext2, "squad": _squad, "pubmedqa": _pubmedqa}[spec](split)


# =============================================================================== splits
def make_splits(cfg, spec):
    """Sample test, then validation, then train; drop from train/val anything sharing a prompt
    or a source group (article / PubMed id) with an earlier split. Saved once, with sha256."""
    name = dataset_name(spec)
    out_dir = os.path.join(cfg["splits_dir"], name)
    if os.path.exists(os.path.join(out_dir, "manifest.json")):
        print(f"{name}: splits exist, reusing {out_dir}")
        return
    rng = random.Random(cfg["seed"])
    blocked_h, blocked_g, manifest = set(), set(), {}
    for split, n in (("test", cfg["test_size"]), ("validation", cfg["val_size"]), ("train", cfg["train_size"])):
        seen, pool = set(), []
        for r in dataset_pool(spec, split):
            h = text_hash(r["prompt"])
            if h in seen or h in blocked_h or r["group"] in blocked_g:
                continue
            seen.add(h)
            pool.append(r)
        if n and len(pool) > n:
            pool = [pool[i] for i in sorted(rng.sample(range(len(pool)), n))]
        blocked_h |= {text_hash(r["prompt"]) for r in pool}
        blocked_g |= {r["group"] for r in pool}
        path = os.path.join(out_dir, f"{split}.jsonl")
        write_jsonl(pool, path)
        manifest[split] = {"n": len(pool), "sha256": sha256(path)}
    write_json(manifest, os.path.join(out_dir, "manifest.json"))
    print(f"{name}: " + ", ".join(f"{k}={v['n']}" for k, v in manifest.items()))


def load_split(cfg, spec, split):
    name = dataset_name(spec)
    path = os.path.join(cfg["splits_dir"], name, f"{split}.jsonl")
    expected = read_json(os.path.join(cfg["splits_dir"], name, "manifest.json"))[split]["sha256"]
    if sha256(path) != expected:
        raise RuntimeError(f"{path} was modified after it was created")
    return read_jsonl(path)


def targets_path(cfg, spec, split):
    return os.path.join(cfg["targets_dir"], teacher_tag(cfg), f"{dataset_name(spec)}_{split}.jsonl")


def training_rows(cfg, split):
    """Split rows with the completion the student learns: teacher output or gold label."""
    rows = []
    for spec in cfg["datasets"]:
        part = load_split(cfg, spec, split)
        limit = cfg["train"].get("train_limit")
        if split == "train" and limit and len(part) > limit:
            part = random.Random(0).sample(part, int(limit))
        if cfg["train"]["targets"] == "teacher":
            targets = {r["id"]: r for r in read_jsonl(targets_path(cfg, spec, split))}
            if cfg["train"].get("drop_truncated"):   # skip answers the teacher did not finish
                n = len(part)
                part = [r for r in part if targets[r["id"]]["stopped"]]
                print(f"{spec}/{split}: dropped {n - len(part)} of {n} truncated teacher answers")
            part = [{**r, "completion": targets[r["id"]]["completion"]} for r in part]
        else:
            part = [{**r, "completion": gold_text(r)} for r in part]
        rows += part
    return rows


def gold_text(r):
    g = r["gold"]
    if r["task"] == "yes_no_maybe":
        return f"{g}. {r.get('long_answer', '')}".strip()
    return g[0] if isinstance(g, list) else str(g)


# =============================================================================== models
def _dtype_kw(dtype):
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def load_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token   # only pads inputs; loss masks never use token ids
    return tok


def load_model(name, precision="bf16", device="cuda"):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if precision in dtypes:
        model = AutoModelForCausalLM.from_pretrained(name, **_dtype_kw(dtypes[precision])).to(device)
    elif precision == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0})
    elif precision == "nf4":
        model = AutoModelForCausalLM.from_pretrained(name, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
    else:  # already-quantised checkpoint (AWQ / GPTQ)
        model = AutoModelForCausalLM.from_pretrained(name, device_map={"": 0}, **_dtype_kw("auto"))
    return model.eval()


def build_student(cfg, teacher, teacher_tok):
    """Return (student, tokenizer, info).

    random:          the teacher's own architecture with `num_layers` fresh blocks; embeddings,
                     final norm and LM head copied from the teacher (D3LM).
    teacher_layers:  same, but the blocks are copied from selected teacher layers (layer pruning).
    pretrained:      an existing small model; for KD it must use the teacher's tokenizer.
    """
    from transformers import AutoModelForCausalLM
    s = cfg["student"]
    info = {"init": s["init"]}
    if s["init"] == "pretrained":
        student = AutoModelForCausalLM.from_pretrained(s["pretrained"], **_dtype_kw(torch.float32))
        tok = load_tokenizer(s["pretrained"])
        if cfg["train"]["objective"] == "kd" and tok.get_vocab() != teacher_tok.get_vocab():
            raise ValueError("Student and teacher tokenizers differ: use train.objective=sft")
    else:
        L = s["num_layers"]
        scfg = copy.deepcopy(teacher.config)
        scfg.num_hidden_layers = L
        if isinstance(getattr(scfg, "layer_types", None), list):
            scfg.layer_types = scfg.layer_types[:L]
        if getattr(scfg, "max_window_layers", None) is not None:
            scfg.max_window_layers = min(scfg.max_window_layers, L)
        if hasattr(scfg, "quantization_config"):
            delattr(scfg, "quantization_config")
        student = AutoModelForCausalLM.from_config(scfg, **_dtype_kw(torch.float32))
        with torch.no_grad():
            student.get_input_embeddings().weight.copy_(teacher.get_input_embeddings().weight.float())
            if getattr(scfg, "tie_word_embeddings", False):
                student.tie_weights()
            else:
                student.get_output_embeddings().weight.copy_(teacher.get_output_embeddings().weight.float())
            student.model.norm.load_state_dict({k: v.float() for k, v in teacher.model.norm.state_dict().items()})
            if s["init"] == "teacher_layers":
                n = teacher.config.num_hidden_layers
                ids = s.get("layer_ids") or sorted({int(round(x)) for x in np.linspace(0, n - 1, L)})
                for j, i in enumerate(ids):
                    student.model.layers[j].load_state_dict(
                        {k: v.float() for k, v in teacher.model.layers[i].state_dict().items()})
                    student.model.layers[j].self_attn.layer_idx = j   # keep the KV cache consistent
                info["teacher_layer_ids"] = ids
        tok = teacher_tok
    if "embeddings" in s["freeze"]:
        student.get_input_embeddings().weight.requires_grad_(False)
    if "lm_head" in s["freeze"]:
        student.get_output_embeddings().weight.requires_grad_(False)
    params = {id(p): p for p in student.parameters()}.values()
    emb = student.get_input_embeddings().weight.numel()
    tied = student.get_output_embeddings().weight is student.get_input_embeddings().weight
    head = 0 if tied else student.get_output_embeddings().weight.numel()
    total = sum(p.numel() for p in params)
    info.update({"num_layers": student.config.num_hidden_layers, "params_total": total,
                 "params_trainable": sum(p.numel() for p in params if p.requires_grad),
                 "params_non_embedding": total - emb - head})
    return student, tok, info


# ======================================================================== tokenisation
def encode(tok, prompt, completion, max_prompt_tokens, max_new_tokens):
    """Prompt and completion are tokenised separately. loss_mask marks completion tokens + EOS.

    The legacy loop masked `labels != pad_id` with pad = eos, which silently dropped every
    EOS target, so students never learned to stop. The mask here never looks at token ids.
    """
    p = tok(prompt)["input_ids"]
    if len(p) > max_prompt_tokens:   # keep BOS and the end of the prompt (the question)
        p = p[:1] + p[-(max_prompt_tokens - 1):] if p[0] == tok.bos_token_id else p[-max_prompt_tokens:]
    completion = completion if completion[:1].isspace() else " " + completion
    c = tok(completion, add_special_tokens=False)["input_ids"][:max_new_tokens] + [tok.eos_token_id]
    return {"input_ids": p + c, "loss_mask": [0] * len(p) + [1] * len(c)}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    out = {k: torch.zeros(len(batch), n, dtype=torch.long) for k in ("input_ids", "attention_mask", "loss_mask")}
    out["input_ids"].fill_(pad_id)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        out["input_ids"][i, :L] = torch.tensor(b["input_ids"])
        out["attention_mask"][i, :L] = 1
        out["loss_mask"][i, :L] = torch.tensor(b["loss_mask"])
    return out


# ========================================================================== generation
@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, batch_size=16, max_prompt_tokens=512, ignore_eos=False, chat=False):
    """Greedy decoding with an explicit GenerationConfig (identical for every model, never the
    checkpoint's own sampling defaults). Returns the continuation only, never the prompt."""
    from transformers import GenerationConfig
    stops = {tok.eos_token_id}
    eos = getattr(model.generation_config, "eos_token_id", None)
    stops |= set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    gc = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, temperature=None, top_p=None, top_k=None,
                          min_new_tokens=max_new_tokens if ignore_eos else None,
                          eos_token_id=None if ignore_eos else sorted(stops), pad_token_id=tok.pad_token_id)
    if chat:   # wrap each prompt in the model's chat template (the template already contains BOS)
        prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                   for p in prompts]
    tok.padding_side, tok.truncation_side = "left", "left"
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_prompt_tokens, add_special_tokens=not chat).to(dev)
        new = model.generate(**enc, generation_config=gc)[:, enc["input_ids"].shape[1]:].tolist()
        for ids in new:
            cut = next((k for k, t in enumerate(ids) if t in stops), None) if not ignore_eos else None
            keep = ids if cut is None else ids[:cut]
            out.append({"text": tok.decode(keep, skip_special_tokens=True),
                        "n_new_tokens": len(ids) if ignore_eos else len(keep) + (cut is not None),
                        "stopped": cut is not None})
    tok.padding_side = "right"
    return out


# ============================================================================== energy
class EnergyMeter:
    """GPU energy from NVML's cumulative energy counter (joules between start() and stop()).

    Measure windows of tens of seconds: NVML power readings are averaged and only partly sampled on
    recent GPUs (arXiv:2312.02741). measure_idle() records idle power so idle energy can be subtracted.
    Without a GPU (or NVML) only the time is reported.
    """

    def __init__(self, enabled=True):
        self.handles, self.idle_w = [], None
        if enabled:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nv = pynvml
                self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            except Exception:
                pass
        self.mode = "counter" if self.handles else "none"

    def _joules(self):
        return sum(self.nv.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles) / 1e3

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        self.e0 = self._joules() if self.handles else 0.0

    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        r = {"seconds": time.perf_counter() - self.t0, "mode": self.mode}
        if self.handles:
            r["energy_j"] = self._joules() - self.e0
            r["kwh"] = r["energy_j"] / 3.6e6
            if self.idle_w is not None:
                r["idle_w"] = self.idle_w
                r["dynamic_energy_j"] = r["energy_j"] - self.idle_w * r["seconds"]
        return r

    def measure_idle(self, seconds):
        if not self.handles or not seconds:
            return None
        self.start()
        time.sleep(seconds)
        self.idle_w = self.stop()["energy_j"] / seconds
        return self.idle_w

    def gpu_info(self):
        names = [self.nv.nvmlDeviceGetName(h) for h in self.handles]
        return [n.decode() if isinstance(n, bytes) else n for n in names]


def parse_args(description, extra=None):
    """Standard CLI: --config file.yaml --set key=value ... plus script-specific arguments."""
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    if extra:
        extra(ap)
    args = ap.parse_args()
    return args, load_config(args.config, args.set)