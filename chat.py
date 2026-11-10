"""Ask the distilled model questions in the terminal.

    python chat.py                                   # best student (runs/ce0.3_L7/best)
    python chat.py --model meta-llama/Llama-3.1-8B-Instruct --chat-template   # the teacher

For each query: paste the abstract/context (or press Enter to skip), then type the question.
Type 'quit' to exit. The prompt uses the same format the student was trained on.
"""
import argparse

import torch

from common import generate, load_model, load_tokenizer

PROMPT = ("Answer the question based on your knowledge. Then give a 1-2 sentence justification.\n\n"
          "Abstract: {context}\nQuestion: {question}\nAnswer:")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/ce0.3_L7/best")
ap.add_argument("--chat-template", action="store_true", help="use for chat models such as the teacher")
ap.add_argument("--max-new-tokens", type=int, default=128)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
tok, model = load_tokenizer(args.model), load_model(args.model, "bf16", device)
n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Loaded {args.model} ({n_params:.2f}B parameters). Type 'quit' to exit.\n")
print(f"Loaded {args.model}. Type 'quit' to exit.\n")

while True:
    context = input("Abstract (Enter to skip): ").strip()
    if context.lower() == "quit":
        break
    question = input("Question: ").strip()
    if question.lower() == "quit":
        break
    prompt = PROMPT.format(context=context or "(none)", question=question)
    out = generate(model, tok, [prompt], args.max_new_tokens, batch_size=1, chat=args.chat_template)[0]
    print(f"\nAnswer: {out['text'].strip()}\n")