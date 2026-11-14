"""Offline smoke test (CPU, tiny random models, no downloads):  python -m pytest -q test_smoke.py

Checks the key fixes (EOS is trained, outputs exclude the prompt, KD loss is 0 for identical
models, a student built from all teacher layers equals the teacher) and runs every script once.
"""
import json
import os
import random
import subprocess
import sys

import pytest
import torch
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only, whatever the GPU / driver setup
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import build_student, encode, generate, load_config, load_model, load_tokenizer  # noqa: E402
from train import kd_loss  # noqa: E402

WORDS = "the a is was cat dog bird fish red blue green black went home today river city king . ? :".split()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast
    d = tmp_path_factory.mktemp("tiny")
    vocab = {w: i for i, w in enumerate(["<unk>", "<s>", "</s>"] + sorted(set(WORDS + ["Question", "Answer", "what", "color"])))}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>")  # no pad token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(LlamaConfig(
        vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False))
    model.save_pretrained(d / "teacher")
    tok.save_pretrained(d / "teacher")
    rng = random.Random(0)
    with open(d / "toy.jsonl", "w") as fh:
        for i in range(150):
            a, c = rng.choice(["cat", "dog", "bird", "fish"]), rng.choice(["red", "blue", "green", "black"])
            fh.write(json.dumps({"prompt": f"the {a} is {c} . the {rng.choice(WORDS[:6])} went home . "
                                           f"Question : what color is the {a} ? Answer :",
                                 "gold": c, "task": "generic", "group": str(i)}) + "\n")
    cfg = {"teacher": str(d / "teacher"), "teacher_precision": "fp32", "teacher_tag": "tiny",
           "datasets": [f"jsonl:{d / 'toy.jsonl'}"], "splits_dir": str(d / "splits"), "targets_dir": str(d / "targets"),
           "train_size": 80, "val_size": 10, "test_size": 10, "max_prompt_tokens": 48, "max_new_tokens": 6,
           "output_dir": str(d / "runs"), "device": "cpu",
           "student": {"num_layers": 2},
           "train": {"max_steps": 6, "eval_every": 3, "gradient_checkpointing": True, "batch_size": 4, "grad_accum": 1, "bf16": False, "lr": 3e-3},
           "energy": {"enabled": False, "idle_seconds": 0}}
    with open(d / "cfg.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    return d


def run(script, *args, cwd):
    subprocess.run([sys.executable, os.path.join(HERE, script), *args], check=True, cwd=cwd)


def test_eos_is_trained_and_kd_loss(tiny):
    tok = load_tokenizer(str(tiny / "teacher"))
    assert tok.pad_token_id == tok.eos_token_id                     # the configuration of the legacy bug
    e = encode(tok, "the cat is red", "the dog", 32, 8)
    assert e["input_ids"][-1] == tok.eos_token_id and e["loss_mask"][-1] == 1
    ids = torch.tensor([e["input_ids"]])
    mask = torch.tensor([e["loss_mask"]])
    logits = torch.randn(1, ids.shape[1], 30)
    loss, kl, _ = kd_loss(logits, logits.clone(), ids, mask, 0.0, 2.0)
    assert abs(float(kl)) < 1e-5 and abs(float(loss)) < 1e-5


def test_skew_kl():
    torch.manual_seed(0)
    ids, mask = torch.randint(0, 30, (2, 8)), torch.ones(2, 8, dtype=torch.long)
    s, t = torch.randn(2, 8, 30, requires_grad=True), torch.randn(2, 8, 30)
    fkl = kd_loss(s, t, ids, mask, 0.0, 1.0, "fkl")[1]
    for div in ("skl", "srkl"):
        assert abs(float(kd_loss(t, t, ids, mask, 0.0, 1.0, div)[1])) < 1e-5           # 0 for identical models
        small = kd_loss(s, t, ids, mask, 0.0, 1.0, div, skew_alpha=1e-6)[1]
        ref = fkl if div == "skl" else kd_loss(s, t, ids, mask, 0.0, 1.0, "rkl")[1]
        assert abs(float(small) - float(ref)) < 1e-3                                     # alpha -> 0 gives fkl / rkl
        assert float(kd_loss(s, t, ids, mask, 0.0, 1.0, div, skew_alpha=0.1)[1]) < float(ref)  # skew is smaller
        loss = kd_loss(s, t, ids, mask, 0.2, 2.0, div)[0]
        loss.backward()
        assert torch.isfinite(s.grad).all()
        s.grad = None


def test_students_and_generation(tiny):
    cfg = load_config(str(tiny / "cfg.yaml"), ["student.init=teacher_layers", "student.num_layers=4"])
    teacher = load_model(str(tiny / "teacher"), "fp32", "cpu")
    tok = load_tokenizer(str(tiny / "teacher"))
    student, _, info = build_student(cfg, teacher, tok)
    x = torch.randint(3, 20, (2, 7))
    with torch.no_grad():
        assert torch.allclose(student(input_ids=x).logits, teacher(input_ids=x).logits, atol=1e-5)
    random_student, _, info = build_student(load_config(str(tiny / "cfg.yaml")), teacher, tok)
    assert info["num_layers"] == 2 and info["params_trainable"] < info["params_total"]
    teacher.generation_config.do_sample = True                       # must be ignored
    outs = generate(teacher, tok, ["the cat is red", "the dog went home today"], 5, batch_size=2)
    assert outs == generate(teacher, tok, ["the cat is red", "the dog went home today"], 5, batch_size=1)
    assert all(not o["text"].startswith("the cat is red") for o in outs)


def test_all_scripts(tiny):
    c = str(tiny / "cfg.yaml")
    run("prepare_data.py", "--config", c, cwd=tiny)
    fast = ["--metrics", "bleu", "rouge1", "rougeL", "chrf", "--metrics-device", "cpu"]   # no model downloads offline
    run("evaluate.py", "--config", c, "--name", "teacher", "--model", str(tiny / "teacher"), "--precision", "fp32",
        *fast, cwd=tiny)
    run("train.py", "--config", c, "--set", "run_name=kd", cwd=tiny)
    run("train.py", "--config", c, "--set", "run_name=sft", "train.objective=sft", cwd=tiny)
    run("train.py", "--config", c, "--layers", "1", "2", "--set", "run_name=depth", cwd=tiny)
    import shutil   # student pruned from another model (student.source), here a copy of the teacher
    shutil.copytree(tiny / "teacher", tiny / "other", dirs_exist_ok=True)
    run("train.py", "--config", c, "--set", "run_name=src_kd", "student.init=teacher_layers",
        f"student.source={tiny / 'other'}", cwd=tiny)
    assert (tiny / "runs" / "src_kd" / "best").exists()
    run("train.py", "--config", c, "--set", "run_name=hid_kd", "student.init=teacher_layers", "train.hidden_weight=1.0", cwd=tiny)
    assert (tiny / "runs" / "hid_kd" / "best").exists()
    s = json.load(open(tiny / "runs" / "kd" / "summary.json"))
    assert s["teacher_used"] and "val_kl" in s["history"][0] and s["student"]["num_layers"] == 2
    assert not json.load(open(tiny / "runs" / "sft" / "summary.json"))["teacher_used"]
    assert os.path.exists(tiny / "runs" / "depth_L1" / "best")
    run("evaluate.py", "--config", c, "--name", "kd", "--model", str(tiny / "runs" / "kd" / "best"),
        "--precision", "fp32", "--reference", "teacher", *fast, cwd=tiny)
    run("evaluate.py", "--config", c, "--name", "floor", "--model", "copy_prompt", "--reference", "teacher", *fast, cwd=tiny)
    m = json.load(open(tiny / "eval" / "kd" / "toy" / "metrics.json"))
    assert {"gold_bleu", "gold_rougeL", "gold_floor_shuffled", "fid_rougeL", "fid_floor_shuffled", "failure"} <= set(m)
    assert {"gold_ppl", "gold_bpc", "fid_ppl", "fid_bpc"} <= set(m) and m["gold_ppl"]["mean"] > 1

    run("evaluate.py", "--config", c, "--name", "kd", "--rescore", "--reference", "teacher", *fast, cwd=tiny)
    m2 = json.load(open(tiny / "eval" / "kd" / "toy" / "metrics.json"))
    assert m2["n"] == m["n"] and "gold_ppl" not in m2            # rescore without --model: no perplexity
    run("evaluate.py", "--config", c, "--name", "kd", "--rescore", "--model", str(tiny / "runs" / "kd" / "best"),
        "--precision", "fp32", "--reference", "teacher", *fast, cwd=tiny)
    m3 = json.load(open(tiny / "eval" / "kd" / "toy" / "metrics.json"))
    assert abs(m3["gold_ppl"]["mean"] - m["gold_ppl"]["mean"]) < 1e-3   # same value without regenerating
    run("energy_benchmark.py", "--config", c, "--name", "kd_hf", "--model", str(tiny / "runs" / "kd" / "best"),
        "--precision", "fp32", "--batch-sizes", "1", "4", "--repeats", "1", "--min-seconds", "0", cwd=tiny)
    b = json.load(open(tiny / "bench" / "kd_hf.json"))
    assert [r["batch_size"] for r in b["hf"]] == [1, 4]
    # break-even with energies filled in by hand (no GPU meter on CPU)
    for name, j in (("teacher", 400.0), ("kd_hf", 40.0)):
        json.dump({"hf": [{"batch_size": 8, "mean": {"j_per_request": j}}]}, open(tiny / "bench" / f"{name}.json", "w"))
    json.dump({"energy": {"energy_j": 3.6e6}}, open(tiny / "train_summary.json", "w"))
    yaml.safe_dump({"baseline": "teacher", "setting": {"batch_size": 8},
                    "options": [{"name": "teacher", "bench": "bench/teacher.json"},
                                {"name": "student", "bench": "bench/kd_hf.json", "train": "train_summary.json"}]},
                   open(tiny / "be.yaml", "w"))
    run("break_even.py", "--spec", "be.yaml", cwd=tiny)
    assert "10000.0" in open(tiny / "results" / "break_even.csv").read()   # 3.6e6 J / 360 J per request


def test_long_answer_and_judge(tmp_path):
    import evaluate as E
    row = {"id": "a", "task": "yes_no_maybe", "gold": "yes", "prompt": "Abstract: ... Question: ...?\nAnswer:",
           "long_answer": "Peer coaching reduced asthma visits."}
    assert E.reference_text(row) == row["long_answer"]
    assert E.answer_body(row, "Yes. Peer coaching reduced visits.") == "Peer coaching reduced visits."
    s = E.text_metrics(["Peer coaching reduced asthma visits."], [row["long_answer"]], ["bleu", "rougeL", "chrf"])
    assert s["rougeL"][0] == pytest.approx(1.0)
    rows = [dict(row, id=str(i)) for i in range(10)]
    preds = [{"id": str(i), "prediction": "yes"} for i in range(10)]
    calls = []
    fake = lambda provider, prompt: calls.append(prompt) or 7.0
    scores = E.run_judge(rows, preds, "fake:m", 4, str(tmp_path / "j.json"), call=fake)
    assert len(scores) == 4 and set(scores.values()) == {7.0} and "Reference answer:\nyes. Peer" in calls[0]
    assert E.run_judge(rows, preds, "fake:m", 4, str(tmp_path / "j.json"), call=fake) == scores and len(calls) == 4  # cached
    again = E.run_judge(list(reversed(rows)), preds, "fake:m", 4, str(tmp_path / "j2.json"), call=fake)
    assert set(again) == set(scores)                                              # same items for every model


def test_chat_template_and_drop_truncated(tiny):
    import common
    tok = load_tokenizer(str(tiny / "teacher"))
    tok.chat_template = "{% for m in messages %}<s>{{ m['content'] }} Answer :{% endfor %}"
    teacher = load_model(str(tiny / "teacher"), "fp32", "cpu")
    out = generate(teacher, tok, ["the cat is red"], 4, chat=True)
    assert len(out) == 1 and out[0]["n_new_tokens"] <= 4
    cfg = load_config(str(tiny / "cfg.yaml"), ["train.drop_truncated=true"])
    if not os.path.exists(common.targets_path(cfg, cfg["datasets"][0], "train")):
        run("prepare_data.py", "--config", str(tiny / "cfg.yaml"), cwd=tiny)
    n_all = len(common.training_rows(load_config(str(tiny / "cfg.yaml")), "train"))
    targets = common.read_jsonl(common.targets_path(cfg, cfg["datasets"][0], "train"))
    assert len(common.training_rows(cfg, "train")) == sum(r["stopped"] for r in targets) <= n_all