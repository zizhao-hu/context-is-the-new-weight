"""Teacher rollout.

For each query Q and context C, runs the model with [C; Q], generates up to
max_new_tokens, and saves the top-k logits at every generated position. These
become the distillation targets for a student that runs without C.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .contexts import build_messages


@dataclass
class Trace:
    ctx_name: str
    query: str
    prompt_with_ctx_ids: torch.Tensor   # (P_ctx,)
    prompt_no_ctx_ids: torch.Tensor     # (P_noctx,)  — what the student sees
    gen_ids: torch.Tensor               # (T,)
    topk_indices: torch.Tensor          # (T, k)
    topk_logprobs: torch.Tensor         # (T, k) — log-softmax over full vocab


@torch.no_grad()
def rollout(
    model,
    tok,
    ctx_name: str,
    query: str,
    max_new_tokens: int = 64,
    top_k: int = 20,
    temperature: float = 0.0,
) -> Trace:
    msgs_ctx = build_messages(ctx_name, query)
    msgs_noctx = build_messages("no_context", query)

    inputs = tok.apply_chat_template(
        msgs_ctx, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    noctx_ids = tok.apply_chat_template(
        msgs_noctx, return_tensors="pt", add_generation_prompt=True
    )[0]

    prompt_len = inputs.shape[1]
    generated = inputs.clone()

    topk_indices: list[torch.Tensor] = []
    topk_logprobs: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        out = model(generated, use_cache=False)
        logits = out.logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)

        topk = torch.topk(logprobs, top_k, dim=-1)
        topk_logprobs.append(topk.values[0].cpu())
        topk_indices.append(topk.indices[0].cpu())

        if temperature == 0.0:
            next_tok = topk.indices[:, :1]
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_tok], dim=-1)
        if next_tok.item() == tok.eos_token_id:
            break

    gen_ids = generated[0, prompt_len:].cpu()
    return Trace(
        ctx_name=ctx_name,
        query=query,
        prompt_with_ctx_ids=inputs[0].cpu(),
        prompt_no_ctx_ids=noctx_ids,
        gen_ids=gen_ids,
        topk_indices=torch.stack(topk_indices),
        topk_logprobs=torch.stack(topk_logprobs),
    )
