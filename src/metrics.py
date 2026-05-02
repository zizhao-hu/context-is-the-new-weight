"""Behavioral equivalence metrics."""
from __future__ import annotations

import torch
from torch.nn.functional import log_softmax, softmax

from .contexts import build_messages


@torch.no_grad()
def per_token_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per position. Both: (T, V)."""
    p = softmax(p_logits, dim=-1)
    log_p = log_softmax(p_logits, dim=-1)
    log_q = log_softmax(q_logits, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


@torch.no_grad()
def teacher_student_logits(teacher, student, tok, ctx_name: str, query: str, max_new_tokens: int = 64):
    """Run teacher with context, student without; greedily decode along the
    teacher's path so logits are aligned position-by-position. Returns
    (teacher_logits, student_logits) of shape (T, V)."""
    msgs_ctx = build_messages(ctx_name, query)
    msgs_noctx = build_messages("no_context", query)

    t_in = tok.apply_chat_template(msgs_ctx, return_tensors="pt", add_generation_prompt=True).to(teacher.device)
    s_in = tok.apply_chat_template(msgs_noctx, return_tensors="pt", add_generation_prompt=True).to(student.device)

    t_logits, s_logits = [], []
    for _ in range(max_new_tokens):
        t_out = teacher(t_in, use_cache=False).logits[:, -1, :]
        s_out = student(s_in, use_cache=False).logits[:, -1, :]
        t_logits.append(t_out[0].cpu())
        s_logits.append(s_out[0].cpu())
        next_tok = t_out.argmax(dim=-1, keepdim=True)
        t_in = torch.cat([t_in, next_tok], dim=-1)
        s_in = torch.cat([s_in, next_tok.to(student.device)], dim=-1)
        if next_tok.item() == tok.eos_token_id:
            break
    return torch.stack(t_logits), torch.stack(s_logits)


@torch.no_grad()
def top1_agreement(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    return (p_logits.argmax(-1) == q_logits.argmax(-1)).float().mean().item()
