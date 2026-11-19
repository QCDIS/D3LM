"""Number of layers and the last layer of a Hugging Face model (no weights are downloaded).

    python model_info.py Qwen/Qwen2.5-7B-Instruct
    python model_info.py runs/ce0.3_L7/best
"""
import sys

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

name = sys.argv[1]
cfg = AutoConfig.from_pretrained(name)
with init_empty_weights():                      # builds the model structure only
    model = AutoModelForCausalLM.from_config(cfg)

print(f"model:            {name}")
print(f"layers:           {cfg.num_hidden_layers}")
print(f"hidden size:      {cfg.hidden_size} neurons per layer output")
print(f"MLP size:         {cfg.intermediate_size} neurons inside each layer's MLP")
print(f"vocabulary:       {cfg.vocab_size} output neurons (LM head)")
print(f"parameters:       {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print("\nlast transformer layer:")
print(model.model.layers[-1])
print("\nfinal norm + output layer:")
print(model.model.norm)
print(model.lm_head)