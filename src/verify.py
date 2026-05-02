"""Phase 3 — behavior verification.

After full-FT distillation, before any expensive analysis: confirm the trained
student actually exhibits the context's behavior on its own training queries.
This is a hard gate. If the student doesn't match the teacher on the training
set, the rest of the analysis is meaningless.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.functional import log_softmax, softmax
from tqdm import tqdm


@torch.no_grad()
def _generate_no_context(model, tok, prompt_ids: list[int], max_new_tokens: int = 96) -> list[int]:
    """Greedy decode from the prompt (no context); return generated token ids."""
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    prompt_len = ids.shape[1]
    for _ in range(max_new_tokens):
        out = model(ids, use_cache=False)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=-1)
        if nxt.item() == tok.eos_token_id:
            break
    return ids[0, prompt_len:].cpu().tolist()


def _exact_match(a: list[int], b: list[int]) -> bool:
    return a == b


def _prefix_match_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@torch.no_grad()
def _per_token_kl(student, base, prompt_ids: list[int], teacher_gen_ids: list[int]) -> float:
    """Average KL between student and base on the teacher's gen path, conditioned on no-ctx prompt.
    Returns scalar mean KL across the answer positions. Useful as a sanity check that the student
    has actually moved away from base."""
    full = torch.tensor(prompt_ids + teacher_gen_ids, dtype=torch.long, device=student.device).unsqueeze(0)
    s_logits = student(full, use_cache=False).logits[0]
    b_logits = base(full, use_cache=False).logits[0]
    # next-token positions = those whose target is in teacher_gen_ids
    pos = torch.arange(len(prompt_ids) - 1, len(prompt_ids) + len(teacher_gen_ids) - 1, device=student.device)
    s = log_softmax(s_logits[pos], dim=-1)
    b = log_softmax(b_logits[pos], dim=-1)
    p_s = s.exp()
    kl_per = (p_s * (s - b)).sum(dim=-1)  # KL(student || base) per pos
    return kl_per.mean().item()


def run(
    student,
    base,
    tok,
    records: list[dict],
    out_path: Path,
    max_new_tokens: int = 96,
    sample_for_kl: int = 25,
) -> dict:
    """Run verification. For each training record:
       - decode student answer (no-context) and compare against teacher answer
       - on a subset, also compute KL(student||base) along teacher's path

    Writes JSON with per-record details + aggregate metrics.
    Returns the aggregate dict for in-process inspection.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n_exact = 0
    n_prefix_at_least_5 = 0
    sum_prefix = 0
    sum_kl = 0.0
    n_kl = 0

    for i, rec in enumerate(tqdm(records, desc="verify")):
        prompt_ids = rec["prompt_no_ctx_ids"]
        teacher_gen = rec["gen_ids"]
        student_gen = _generate_no_context(student, tok, prompt_ids, max_new_tokens=max_new_tokens)

        em = _exact_match(student_gen, teacher_gen)
        pm = _prefix_match_len(student_gen, teacher_gen)

        kl = None
        if i < sample_for_kl:
            try:
                kl = _per_token_kl(student, base, prompt_ids, teacher_gen)
                sum_kl += kl
                n_kl += 1
            except Exception as e:  # noqa: BLE001
                kl = None

        n_exact += int(em)
        n_prefix_at_least_5 += int(pm >= 5)
        sum_prefix += pm

        rows.append({
            "query": rec["query"],
            "teacher_answer": rec["answer_text"],
            "student_answer": tok.decode(student_gen, skip_special_tokens=True),
            "exact_match": em,
            "prefix_match_len": pm,
            "kl_student_vs_base": kl,
        })

    agg = {
        "n_total": len(records),
        "n_exact_match": n_exact,
        "exact_match_rate": n_exact / max(1, len(records)),
        "n_prefix_match_ge_5": n_prefix_at_least_5,
        "prefix_match_rate_ge_5": n_prefix_at_least_5 / max(1, len(records)),
        "mean_prefix_match_len": sum_prefix / max(1, len(records)),
        "mean_kl_student_vs_base": (sum_kl / n_kl) if n_kl else None,
    }

    out_path.write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=2), encoding="utf-8")
    return agg
