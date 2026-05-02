"""HF model loading."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_name: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    return model, tok
