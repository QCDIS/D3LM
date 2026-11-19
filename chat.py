"""Ask a model questions in the terminal.

    python chat.py                                   # best PubMedQA student (runs/ce0.3_L7/best)
    python chat.py --model meta-llama/Llama-3.1-8B-Instruct --chat-template   # the teacher
    python chat.py --code --model Qwen/Qwen2.5-Coder-1.5B-Instruct --chat-template     # write Python code
    python chat.py --code --model runs/code_qwen25coder-1.5b_kd/best --test 0          # test problem 0, runs its tests

PubMedQA mode: paste the abstract (or press Enter to skip), then type the question.
Code mode: type a programming task. With --test N, the model solves test problem N (0-541) and its tests are run.
Type 'quit' to exit. Prompts use the same format the students were trained on.
"""
import argparse
import json

import torch

from common import CODE_INSTRUCTION, generate, load_model, load_tokenizer

PROMPT = ("Answer the question based on your knowledge. Then give a 1-2 sentence justification.\n\n"
          "Abstract: {context}\nQuestion: {question}\nAnswer:")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/ce0.3_L7/best")
ap.add_argument("--chat-template", action="store_true", help="use for chat models such as the teacher")
ap.add_argument("--code", action="store_true", help="Python code mode")
ap.add_argument("--test", type=int, default=None, help="code mode: solve test problem N and run its tests")
ap.add_argument("--max-new-tokens", type=int, default=None, help="default: 512 for code, 128 otherwise")
args = ap.parse_args()
max_new = args.max_new_tokens or (512 if args.code else 128)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok, model = load_tokenizer(args.model), load_model(args.model, "bf16", device)
n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Loaded {args.model} ({n_params:.2f}B parameters) on {device}. Type 'quit' to exit.\n")


def ask(prompt):
    return generate(model, tok, [prompt], max_new, batch_size=1, chat=args.chat_template)[0]["text"].strip()


if args.code and args.test is not None:   # one test problem from the code test split, with its tests
    from evaluate import task_scores
    row = [json.loads(l) for l in open("data/splits/code/test.jsonl")][args.test]
    print(f"=== {row['id']} ===\n{row['prompt']}\n")
    answer = ask(row["prompt"])
    print(f"=== answer ===\n{answer}\n")
    print("=== tests:", task_scores(row, answer)["result"], "===")
    raise SystemExit

while True:
    if args.code:
        task = input("Task: ").strip()
        if task.lower() == "quit":
            break
        prompt = CODE_INSTRUCTION + task
    else:
        context = input("Abstract (Enter to skip): ").strip()
        if context.lower() == "quit":
            break
        question = input("Question: ").strip()
        if question.lower() == "quit":
            break
        prompt = PROMPT.format(context=context or "(none)", question=question)
    print(f"\nAnswer:\n{ask(prompt)}\n")