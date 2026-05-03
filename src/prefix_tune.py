"""Prefix tuning: per-layer learnable K/V prefix, rest of model frozen.

The hypothesis we test: a small per-layer K/V prefix that the model attends
to on every forward pass is a more faithful "compression" of context than
full fine-tuning. Full FT changes how the model computes; prefix tuning
preserves the computation and adds extra keys/values for attention to read,
which is mechanistically much closer to what prepending the context did
in-context.

We use HuggingFace PEFT's PrefixTuningConfig which generates per-layer
K/V from a shared learned embedding through a small projection. The
resulting trainable parameter count is on the order of
  n_layers * n_virtual_tokens * 2 * hidden_size
which is ~500K--2M parameters for typical settings — vs 8B for full FT.

Train with the same next-token CE loss as full FT (Setting 2 of the paper)
on the synthetic (Q, A) pairs generated under context.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from peft import PrefixTuningConfig, TaskType, get_peft_model
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


def make_prefix_model(model, num_virtual_tokens: int = 16, prefix_projection: bool = False,
                     encoder_hidden_size: int | None = None):
    """Wrap a base model with PEFT prefix tuning. All base weights frozen.

    `prefix_projection=False` (the default here) gives DIRECT per-layer K/V
    parameters of size num_layers * num_virtual_tokens * 2 * hidden_size,
    typically ~tens of millions of parameters. With prefix_projection=True
    a small MLP from a shared embedding produces the per-layer K/V; this
    is much heavier (~10x more params) and harder to optimise on small
    training sets, so we default to direct.
    """
    if not _HAS_PEFT:
        raise RuntimeError("peft is not installed — `uv pip install peft`")
    cfg = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,
        prefix_projection=prefix_projection,
        encoder_hidden_size=encoder_hidden_size or model.config.hidden_size,
    )
    return get_peft_model(model, cfg)


def train_prefix(
    model,  # PEFT-wrapped, base frozen
    tok,
    records: list[dict],
    *,
    lr: float = 5e-3,           # higher than full FT — only tiny prefix params
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
    use_8bit_optim: bool = True,
    log_every: int = 10,
    device: str = "cuda",
):
    """Train ONLY the prefix params. All base weights frozen by PEFT."""
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[prefix] trainable parameters: {n_trainable:,}")

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


def save_prefix(model, out_dir: str | Path):
    """PEFT save_pretrained writes only the adapter. Tiny — KB to MB scale."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)


def load_prefix(base_model, prefix_dir: str | Path):
    """Reload a PEFT-prefix model from disk. Returns model with prefix attached
    and base frozen."""
    if not _HAS_PEFT:
        raise RuntimeError("peft is not installed")
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, str(prefix_dir))
