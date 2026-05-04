"""Phase 3x — held-out ROUGE validation for any PEFT-adapter model.

Same as phase3c_rouge_validation.py but loads a PEFT adapter (lora / prompt /
prefix) as the student instead of a full-FT save_dir. Saves to
outputs/01_synth_distill_kvdw/<method>_<context>/rouge_validation.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src import models, peft_adapter, rouge_eval, use_cases, validation_queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--method", required=True, choices=["lora", "prompt", "prefix"])
    ap.add_argument("--n-queries", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_context = args.context

    adapter_dir = Path(cfg["saves_root"]) / f"{args.method}_{base_context}"
    out_dir = Path(cfg["out_root"]) / f"{args.method}_{base_context}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rouge_validation.json"

    if base_context == "factual":
        val_queries = validation_queries.sample_validation(
            total=args.n_queries, recall_facts=use_cases.FACTUAL_KEYS,
        )
    else:
        val_queries = validation_queries.sample_validation(total=args.n_queries)

    print(f"[phase3x:{args.method}] context={args.context}  n_queries={len(val_queries)}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base, tok = models.load(cfg["model"], dtype=dtype, device=args.device)
    student = peft_adapter.load_adapter(base, adapter_dir)
    student.eval()

    means = rouge_eval.evaluate(
        base, student, tok, base_context, val_queries, out_path,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n[phase3x:{args.method}] aggregate:")
    print(f"  R(base, ctx)    = {means['rougeL_base_ctx_mean']:.3f}")
    print(f"  R(adapter, ctx) = {means['rougeL_stu_ctx_mean']:.3f}")
    print(f"  R(adapter, base)= {means['rougeL_stu_base_mean']:.3f}")
    print(f"  gap_closure_L   = {means['gap_closure_rougeL']:.3f}")


if __name__ == "__main__":
    main()
