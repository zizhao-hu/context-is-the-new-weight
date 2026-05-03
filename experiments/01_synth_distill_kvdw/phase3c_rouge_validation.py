"""Phase 3c — held-out ROUGE validation.

For one context, generate base/with-context/student answers on a held-out
validation set and compute ROUGE between all three pairs. Reports gap-closure:
how much of the base→with-context distance the student covers.

This is the gate before phase4 KV-vs-ΔW analysis. If gap closure is small,
the FT student hasn't actually learned the context and the activation
analysis isn't meaningful.

Usage:
  python experiments/01_synth_distill_kvdw/phase3c_rouge_validation.py \
    --config experiments/01_synth_distill_kvdw/config.yaml \
    --context haiku
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src import models, rouge_eval, use_cases, validation_queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True,
                    help="Context name. Use ctxonly_<name> to validate Setting 3 students.")
    ap.add_argument("--n-queries", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # The student lives at saves/<args.context>; the base v_ctx is computed
    # against the ORIGINAL context, not the ctxonly_ alias.
    base_context = args.context
    if args.context.startswith("ctxonly_"):
        base_context = args.context[len("ctxonly_"):]

    save_dir = Path(cfg["saves_root"]) / args.context
    out_dir = Path(cfg["out_root"]) / args.context
    out_path = out_dir / "rouge_validation.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build held-out validation queries — disjoint from training pool.
    if base_context == "factual":
        val_queries = validation_queries.sample_validation(
            total=args.n_queries, recall_facts=use_cases.FACTUAL_KEYS,
        )
    else:
        val_queries = validation_queries.sample_validation(total=args.n_queries)

    print(f"[phase3c] context={args.context} (base_context={base_context})")
    print(f"[phase3c] n_queries={len(val_queries)} student={save_dir}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base, tok = models.load(cfg["model"], dtype=dtype, device=args.device)
    student, _ = models.load(str(save_dir), dtype=dtype, device=args.device)
    base.eval()
    student.eval()

    means = rouge_eval.evaluate(
        base, student, tok, base_context, val_queries, out_path,
        max_new_tokens=args.max_new_tokens,
    )

    # Print headline numbers
    print("\n[phase3c] aggregate results:")
    print(f"  R(base, ctx)    = {means['rougeL_base_ctx_mean']:.3f}  (reference gap, lower means base differs more from ctx)")
    print(f"  R(student, ctx) = {means['rougeL_stu_ctx_mean']:.3f}  (success metric, higher = closer to ctx)")
    print(f"  R(student, base)= {means['rougeL_stu_base_mean']:.3f}  (how far student moved from base)")
    print(f"  gap_closure_L   = {means['gap_closure_rougeL']:.3f}  (1.0 = student matches ctx; 0.0 = student is base)")

    # Pass/fail signal for the orchestrator
    if means["gap_closure_rougeL"] >= 0.40:
        print(f"[phase3c] STATUS=PASS  (gap_closure_rougeL={means['gap_closure_rougeL']:.3f} >= 0.40)")
    elif means["gap_closure_rougeL"] >= 0.20:
        print(f"[phase3c] STATUS=WEAK  (gap_closure_rougeL={means['gap_closure_rougeL']:.3f} in [0.20, 0.40))")
    else:
        print(f"[phase3c] STATUS=FAIL  (gap_closure_rougeL={means['gap_closure_rougeL']:.3f} < 0.20)")


if __name__ == "__main__":
    main()
