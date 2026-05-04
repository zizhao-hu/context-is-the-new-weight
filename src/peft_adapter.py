"""Unified PEFT adapter wrapper covering LoRA, prompt tuning, and prefix tuning.

We test the hypothesis that *some* parameter-efficient compression of context
into the weights might preserve general capability where full fine-tuning
destroys it. We compare three families:

- prefix tuning  : per-layer learnable K/V (mechanistically closest to leaving C
                   in the KV cache; covered separately in src/prefix_tune.py)
- LoRA           : low-rank ΔW on attention projections — the canonical PEFT
- prompt tuning  : input-layer soft prompt only — the most "input-side"
                   compression of C, mechanistically closest to "C as a token
                   sequence the model attends to once at the input"

All three are PEFT adapters trained with the same next-token CE loss on the
synthetic (Q, A) pairs used for full-FT distillation.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from peft import (
        LoraConfig,
        PeftModel,
        PrefixTuningConfig,
        PromptTuningConfig,
        TaskType,
        get_peft_model,
    )
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False

try:
    from bitsandbytes.optim import AdamW8bit as _AdamW8bit
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False
    _AdamW8bit = None

from .distill import QADataset, collate_qa


DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def make_adapter(
    model,
    method: str,
    *,
    num_virtual_tokens: int = 16,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_targets: list[str] | None = None,
):
    """Wrap a base model with a PEFT adapter of the requested family.

    All base weights are frozen by PEFT. Only the adapter parameters are
    trainable.

    method:
      - "lora"   : low-rank ΔW on attention projections (q,k,v,o).
      - "prompt" : input-layer soft prompt of `num_virtual_tokens` virtual
                   tokens. Most input-side; mechanistically closest to "C as
                   a tokenised input the model reads once."
      - "prefix" : per-layer K/V prefix. Direct (no projection), so trainable
                   parameters scale with n_layers.
    """
    if not _HAS_PEFT:
        raise RuntimeError("peft is not installed — `uv pip install peft`")

    if method == "prefix":
        cfg = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
            prefix_projection=False,
            encoder_hidden_size=model.config.hidden_size,
        )
    elif method == "lora":
        targets = lora_targets or DEFAULT_LORA_TARGETS
        cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=targets,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    elif method == "prompt":
        cfg = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
        )
    else:
        raise ValueError(f"unknown method: {method!r} (expected lora/prompt/prefix)")

    return get_peft_model(model, cfg)


def train_adapter(
    model,
    tok,
    records: list[dict],
    *,
    lr: float,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
    use_8bit_optim: bool = True,
    log_every: int = 10,
    device: str = "cuda",
):
    """Train ONLY the adapter params on (Q, A) pairs with next-token CE loss.

    Identical objective to full-FT distillation; only the parameter set differs.
    Caller picks lr appropriate to the method (LoRA ~3e-4; prompt/prefix ~1e-2).
    """
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[peft] trainable parameters: {n_trainable:,}")

    optim_cls = _AdamW8bit if (use_8bit_optim and _HAS_BNB) else torch.optim.AdamW
    optim = optim_cls(trainable, lr=lr)

    ds = QADataset(records)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_qa(b, pad_id=tok.pad_token_id),
    )

    step = 0
    optim.zero_grad()
    for epoch in range(epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            step += 1
            if step % grad_accum == 0:
                optim.step()
                optim.zero_grad()
            if step % log_every == 0:
                yield {"epoch": epoch, "step": step, "loss": loss.item() * grad_accum,
                       "n_trainable": n_trainable}
        if step % grad_accum != 0:
            optim.step()
            optim.zero_grad()


def save_adapter(model, out_dir: str | Path):
    """PEFT save_pretrained writes only the adapter (KB to tens of MB)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)


def load_adapter(base_model, adapter_dir: str | Path):
    """Reload any PEFT adapter (lora/prompt/prefix) from disk on a fresh
    base model. PEFT auto-detects the adapter type from the saved config."""
    if not _HAS_PEFT:
        raise RuntimeError("peft is not installed")
    return PeftModel.from_pretrained(base_model, str(adapter_dir))
