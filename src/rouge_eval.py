"""ROUGE-based held-out validation.

For each held-out query Q', generate three answers:
  A_base    = base model on Q' with no context
  A_ctx     = base model on [C; Q'] (the in-context teacher target)
  A_student = trained student (θ_C) on Q' with no context

Then compute three ROUGE pairs:
  R(base, ctx)    — reference gap between no-context base and with-context teacher
  R(student, ctx) — how close the student is to the in-context target
  R(student, base) — how far the student moved from base

The headline metric is gap closure:

  gap_closure = (R(student, ctx) - R(base, ctx)) / (1 - R(base, ctx))

A value of 0.0 means the FT didn't move toward the target at all; 1.0 means it
fully matched the target. Negative means it moved the wrong way.

This must show meaningful gap closure on a held-out set BEFORE running the
KV-vs-ΔW activation analysis — otherwise the analysis is on a student that
hasn't actually learned the context.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from rouge_score import rouge_scorer

from . import contexts as ctx_lib


def _chat_ids(tok, messages) -> torch.Tensor:
    out = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if isinstance(out, torch.Tensor):
        return out
    return out["input_ids"]


@torch.no_grad()
def _generate(model, tok, ctx_name: str, query: str, max_new_tokens: int = 96) -> str:
    msgs = ctx_lib.build_messages(ctx_name, query)
    ids = _chat_ids(tok, msgs).to(model.device)
    prompt_len = ids.shape[1]
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def evaluate(
    base,
    student,
    tok,
    ctx_name: str,
    val_queries: list[str],
    out_path: Path,
    max_new_tokens: int = 96,
) -> dict:
    """Run the full base/ctx/student generation triple and ROUGE comparison.

    Writes per-query rows + aggregate to `out_path` and returns the aggregate dict.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, q in enumerate(val_queries):
        a_base = _generate(base, tok, "no_context", q, max_new_tokens)
        a_ctx = _generate(base, tok, ctx_name, q, max_new_tokens)
        a_stu = _generate(student, tok, "no_context", q, max_new_tokens)

        s_base_ctx = scorer.score(a_ctx, a_base)
        s_stu_ctx = scorer.score(a_ctx, a_stu)
        s_stu_base = scorer.score(a_base, a_stu)

        rows.append({
            "query": q,
            "a_base": a_base[:300],
            "a_ctx": a_ctx[:300],
            "a_student": a_stu[:300],
            "rouge1_base_ctx": s_base_ctx["rouge1"].fmeasure,
            "rouge2_base_ctx": s_base_ctx["rouge2"].fmeasure,
            "rougeL_base_ctx": s_base_ctx["rougeL"].fmeasure,
            "rouge1_stu_ctx": s_stu_ctx["rouge1"].fmeasure,
            "rouge2_stu_ctx": s_stu_ctx["rouge2"].fmeasure,
            "rougeL_stu_ctx": s_stu_ctx["rougeL"].fmeasure,
            "rouge1_stu_base": s_stu_base["rouge1"].fmeasure,
            "rougeL_stu_base": s_stu_base["rougeL"].fmeasure,
        })
        if (i + 1) % 5 == 0:
            print(f"  [{ctx_name}] {i+1}/{len(val_queries)}  R(base,ctx)={s_base_ctx['rougeL'].fmeasure:.3f}  R(stu,ctx)={s_stu_ctx['rougeL'].fmeasure:.3f}")

    n = len(rows)
    means = {f"{k}_mean": sum(r[k] for r in rows) / n for k in [
        "rouge1_base_ctx", "rouge2_base_ctx", "rougeL_base_ctx",
        "rouge1_stu_ctx", "rouge2_stu_ctx", "rougeL_stu_ctx",
        "rouge1_stu_base", "rougeL_stu_base",
    ]}

    # Gap closure for each ROUGE flavor
    for flavor in ("rouge1", "rouge2", "rougeL"):
        base_ctx = means[f"{flavor}_base_ctx_mean"]
        stu_ctx = means[f"{flavor}_stu_ctx_mean"]
        means[f"gap_closure_{flavor}"] = (stu_ctx - base_ctx) / max(1e-6, (1.0 - base_ctx))

    means["n_queries"] = n
    means["context"] = ctx_name

    out_path.write_text(json.dumps({"aggregate": means, "rows": rows}, indent=2), encoding="utf-8")
    return means
