"""Student distillation.

Train the student to reproduce the teacher's behavior, conditioning ONLY on the
no-context prompt + the previously sampled tokens. Two paths:

- Full fine-tune with next-token CE on the teacher's generated tokens
  (`train_full_ft` — the sprint default).
- LoRA + KL on top-k teacher logits (`train_step` — kept for fast iteration /
  ablation; not used in the sprint).

The hypothesis: the resulting ΔW is functionally equivalent to the original
context C.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False

try:
    from bitsandbytes.optim import AdamW8bit as _AdamW8bit
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False
    _AdamW8bit = None


@dataclass
class DistillExample:
    input_ids: torch.Tensor      # (L,)  — prompt_no_ctx + gen_ids[:-1]
    gen_start: int               # index where generated tokens begin
    topk_indices: torch.Tensor   # (T, k)
    topk_logprobs: torch.Tensor  # (T, k)  — full-vocab log-softmax, top-k slice


def build_examples(traces) -> list[DistillExample]:
    """Concatenate no-ctx prompt with the teacher's generated tokens.

    The student sees the no-ctx prompt as input. We supervise the next-token
    distribution at each position from gen_start onward.
    """
    out = []
    for tr in traces:
        gen_start = tr.prompt_no_ctx_ids.shape[0]
        # Input: [prompt_no_ctx; gen_ids]. The model predicts position i+1
        # from positions 0..i, so to supervise gen position t (0-indexed within
        # gen) we look at the model output at index gen_start + t - 1.
        input_ids = torch.cat([tr.prompt_no_ctx_ids, tr.gen_ids], dim=0)
        out.append(
            DistillExample(
                input_ids=input_ids,
                gen_start=gen_start,
                topk_indices=tr.topk_indices,
                topk_logprobs=tr.topk_logprobs,
            )
        )
    return out


class DistillDataset(Dataset):
    def __init__(self, examples: list[DistillExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch: list[DistillExample], pad_id: int):
    max_len = max(ex.input_ids.shape[0] for ex in batch)
    max_T = max(ex.topk_indices.shape[0] for ex in batch)
    k = batch[0].topk_indices.shape[1]
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long)
    topk_idx = torch.zeros((B, max_T, k), dtype=torch.long)
    topk_lp = torch.full((B, max_T, k), -1e9)
    target_pos = torch.full((B, max_T), -1, dtype=torch.long)  # absolute pos in input_ids
    target_mask = torch.zeros((B, max_T), dtype=torch.bool)

    for i, ex in enumerate(batch):
        L = ex.input_ids.shape[0]
        input_ids[i, :L] = ex.input_ids
        attn_mask[i, :L] = 1
        T = ex.topk_indices.shape[0]
        topk_idx[i, :T] = ex.topk_indices
        topk_lp[i, :T] = ex.topk_logprobs
        # supervise output at index gen_start + t - 1 for t = 0..T-1
        for t in range(T):
            target_pos[i, t] = ex.gen_start + t - 1
        target_mask[i, :T] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "topk_idx": topk_idx,
        "topk_lp": topk_lp,
        "target_pos": target_pos,
        "target_mask": target_mask,
    }


def kl_topk(student_logits: torch.Tensor, teacher_topk_lp: torch.Tensor, teacher_topk_idx: torch.Tensor) -> torch.Tensor:
    """
    student_logits:   (N, V)  — gathered at supervised positions
    teacher_topk_lp:  (N, k)  — full-vocab log-softmax at top-k slice
    teacher_topk_idx: (N, k)
    Returns scalar forward KL renormalized over the teacher's top-k support.
    """
    teacher_p = teacher_topk_lp.softmax(dim=-1)                     # (N, k)
    student_lp = log_softmax(student_logits, dim=-1)                # (N, V)
    student_lp_k = student_lp.gather(-1, teacher_topk_idx)          # (N, k)
    # renormalize student over the top-k support
    student_lp_k = student_lp_k - student_lp_k.logsumexp(dim=-1, keepdim=True)

    teacher_lp_k = teacher_p.clamp_min(1e-12).log()
    kl = (teacher_p * (teacher_lp_k - student_lp_k)).sum(dim=-1)
    return kl.mean()


def make_lora(model, r: int = 16, alpha: int = 32, target_modules=None):
    if not _HAS_PEFT:
        raise RuntimeError("peft is not installed — `uv pip install peft`")
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


# ----------------------------------------------------------------------------
# Full fine-tune path (sprint default).
# ----------------------------------------------------------------------------


@dataclass
class QAExample:
    """A single (prompt, answer) example for next-token CE.

    `input_ids`: full sequence = prompt_no_ctx + gen_ids.
    `prompt_len`: index where the answer starts (CE is masked before this).
    """
    input_ids: torch.Tensor
    prompt_len: int


class QADataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i) -> QAExample:
        rec = self.records[i]
        prompt_ids = torch.tensor(rec["prompt_no_ctx_ids"], dtype=torch.long)
        gen_ids = torch.tensor(rec["gen_ids"], dtype=torch.long)
        input_ids = torch.cat([prompt_ids, gen_ids], dim=0)
        return QAExample(input_ids=input_ids, prompt_len=prompt_ids.shape[0])


def collate_qa(batch: list[QAExample], pad_id: int):
    max_len = max(ex.input_ids.shape[0] for ex in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, ex in enumerate(batch):
        L = ex.input_ids.shape[0]
        input_ids[i, :L] = ex.input_ids
        attn[i, :L] = 1
        # CE is supervised on the answer tokens only.
        labels[i, ex.prompt_len:L] = ex.input_ids[ex.prompt_len:]
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def train_full_ft(
    model,
    tok,
    records: list[dict],
    *,
    lr: float = 2e-5,
    epochs: int = 2,
    batch_size: int = 1,
    grad_accum: int = 8,
    use_8bit_optim: bool = True,
    enable_grad_ckpt: bool = True,
    log_every: int = 10,
    device: str = "cuda",
):
    """Full fine-tune `model` on (Q -> A) pairs using next-token CE.

    Yields per-step log dicts: {epoch, step, loss}. Caller decides where to
    print / write.
    """
    if enable_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        # gradient checkpointing requires use_cache=False; HF handles it.
        if hasattr(model, "config"):
            model.config.use_cache = False

    model.train()

    optim_cls = _AdamW8bit if (use_8bit_optim and _HAS_BNB) else torch.optim.AdamW
    optim = optim_cls(model.parameters(), lr=lr)

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
        for batch in tqdm(dl, desc=f"full-ft epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            step += 1
            if step % grad_accum == 0:
                optim.step()
                optim.zero_grad()
            if step % log_every == 0:
                yield {"epoch": epoch, "step": step, "loss": loss.item() * grad_accum}
        # flush remaining gradients at epoch end
        if step % grad_accum != 0:
            optim.step()
            optim.zero_grad()


def save_full_ft(model, tok, out_dir: str | Path):
    """Save trained model + tokenizer via HF `save_pretrained`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


# ----------------------------------------------------------------------------
# Setting 3: train on the context string itself (no Q, A)
# ----------------------------------------------------------------------------


def train_context_only(
    model,
    tok,
    ctx_name: str,
    *,
    n_steps: int = 100,
    lr: float = 2e-5,
    grad_accum: int = 8,
    use_8bit_optim: bool = True,
    enable_grad_ckpt: bool = True,
    log_every: int = 10,
    device: str = "cuda",
):
    """Standard language-modeling fine-tune on the chat-template-formatted
    context string only (system prompt + any few-shot demos), no Q/A pairs.

    The model sees the same short sequence at every step. CE loss applied to
    all tokens. Yields per-step log dicts.
    """
    from .contexts import CONTEXTS

    ctx = CONTEXTS[ctx_name]
    msgs: list[dict] = []
    if ctx.system:
        msgs.append({"role": "system", "content": ctx.system})
    for shot in ctx.shots:
        msgs.append({"role": "user", "content": shot["user"]})
        msgs.append({"role": "assistant", "content": shot["assistant"]})
    if not msgs:
        raise ValueError(f"Context {ctx_name} has no system prompt or shots to train on.")

    text_ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=False)
    if not isinstance(text_ids, torch.Tensor):
        text_ids = text_ids["input_ids"]
    input_ids = text_ids.to(device)
    n_tokens = int(input_ids.shape[-1])

    if enable_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model.train()

    optim_cls = _AdamW8bit if (use_8bit_optim and _HAS_BNB) else torch.optim.AdamW
    optim = optim_cls(model.parameters(), lr=lr)
    optim.zero_grad()

    for step in range(1, n_steps + 1):
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss / grad_accum
        loss.backward()
        if step % grad_accum == 0:
            optim.step()
            optim.zero_grad()
        if step % log_every == 0:
            yield {"step": step, "loss": loss.item() * grad_accum, "n_tokens": n_tokens}
    # flush trailing gradients
    if n_steps % grad_accum != 0:
        optim.step()
        optim.zero_grad()


def train_step(model, batch, optimizer):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    logits = out.logits  # (B, L, V)

    # Gather logits at supervised positions
    B, T = batch["target_pos"].shape
    pos = batch["target_pos"].clamp_min(0)  # negatives masked below
    flat_logits = logits.gather(
        1, pos.unsqueeze(-1).expand(-1, -1, logits.size(-1))
    )  # (B, T, V)

    mask = batch["target_mask"]
    valid_logits = flat_logits[mask]            # (N, V)
    valid_topk_lp = batch["topk_lp"][mask]      # (N, k)
    valid_topk_idx = batch["topk_idx"][mask]    # (N, k)

    loss = kl_topk(valid_logits, valid_topk_lp, valid_topk_idx)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
