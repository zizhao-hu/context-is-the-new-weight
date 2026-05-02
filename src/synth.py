"""Phase 1 — synthetic dataset generation.

For a fixed context `C`, sample queries from the use-case bank, run the base
model with `[C; Q]`, and store the resulting `(Q, A, top-k logits)` triples.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from . import contexts as ctx_lib
from . import teacher as teacher_mod
from . import use_cases


def _trace_to_record(trace, ctx_name: str, tok) -> dict:
    """Convert a teacher.Trace into a JSON-serializable dict."""
    return {
        "context": ctx_name,
        "query": trace.query,
        "answer_text": tok.decode(trace.gen_ids, skip_special_tokens=True),
        "prompt_no_ctx_ids": trace.prompt_no_ctx_ids.tolist(),
        "gen_ids": trace.gen_ids.tolist(),
        "topk_indices": trace.topk_indices.tolist(),
        "topk_logprobs": trace.topk_logprobs.tolist(),
    }


def generate(
    model,
    tok,
    ctx_name: str,
    n_queries: int,
    out_path: Path,
    max_new_tokens: int = 96,
    top_k: int = 20,
    temperature: float = 0.0,
    seed: int = 0,
) -> int:
    """Generate `n_queries` triples for context `ctx_name`. Returns count
    actually written after curation. Streams to JSONL as it goes."""
    if ctx_name == "factual":
        queries = use_cases.sample_queries(
            seed=seed, total=n_queries, recall_facts=use_cases.FACTUAL_KEYS
        )
    else:
        queries = use_cases.sample_queries(seed=seed, total=n_queries)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for q in tqdm(queries, desc=f"phase1[{ctx_name}]"):
            trace = teacher_mod.rollout(
                model, tok, ctx_name, q,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                temperature=temperature,
            )
            rec = _trace_to_record(trace, ctx_name, tok)
            if not _curate(rec):
                continue
            f.write(json.dumps(rec) + "\n")
            written += 1
    return written


def _curate(rec: dict) -> bool:
    """Drop garbage rows. Conservative filters: empty answer, model said
    'I cannot' / refusal, answer too short."""
    text = (rec.get("answer_text") or "").strip()
    if len(text) < 3:
        return False
    refusals = ("i cannot", "i can't", "i'm sorry, but i can't", "i don't have")
    low = text.lower()
    if any(low.startswith(r) for r in refusals):
        return False
    if len(rec.get("gen_ids", [])) < 2:
        return False
    return True


def load_dataset(path: Path) -> list[dict]:
    """Load a synth dataset JSONL into a list of records."""
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
