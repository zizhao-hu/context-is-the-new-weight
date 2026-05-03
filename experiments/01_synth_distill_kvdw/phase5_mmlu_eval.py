"""Phase 5 — MMLU-Pro accuracy for one context across three model variants.

For one context $C$, run the same MMLU-Pro question subset through:
  base       — base model, no context
  in_context — base model with C prepended at inference
  distilled  — context-distilled student, no context

Save accuracy + per-question rows to:
  outputs/01_synth_distill_kvdw/<context>/mmlu_pro_eval.json

Usage:
  python experiments/01_synth_distill_kvdw/phase5_mmlu_eval.py \
    --config experiments/01_synth_distill_kvdw/config.yaml \
    --context haiku --eval-path data/mmlu_pro_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src import eval_mmlu, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--eval-path", default="data/mmlu_pro_eval.jsonl")
    ap.add_argument("--n-questions", type=int, default=300,
                    help="Cap eval set size (for time-bounded runs)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # The student lives at saves/<context>; we always evaluate the in-context
    # variant against the unaliased context name.
    base_context = args.context
    if args.context.startswith("ctxonly_"):
        base_context = args.context[len("ctxonly_"):]

    save_dir = Path(cfg["saves_root"]) / args.context
    out_dir = Path(cfg["out_root"]) / args.context
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mmlu_pro_eval.json"

    eval_path = Path(args.eval_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"MMLU-Pro eval set not found at {eval_path} — run scripts/download_mmlu_pro.py first")
    dataset = eval_mmlu.load_jsonl(eval_path)[: args.n_questions]
    print(f"[phase5] context={args.context}  n_questions={len(dataset)}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    print("[phase5] loading base model")
    base, tok = models.load(cfg["model"], dtype=dtype, device=args.device)
    base.eval()

    print("[phase5] eval base no-context")
    base_result = eval_mmlu.evaluate(base, tok, "no_context", dataset)
    print(f"  base accuracy: {base_result['accuracy']:.3f}  ({base_result['n_correct']}/{base_result['n_total']})")

    print(f"[phase5] eval base + {base_context} context")
    in_ctx_result = eval_mmlu.evaluate(base, tok, base_context, dataset)
    print(f"  in_context accuracy: {in_ctx_result['accuracy']:.3f}  ({in_ctx_result['n_correct']}/{in_ctx_result['n_total']})")

    del base
    torch.cuda.empty_cache()

    print(f"[phase5] loading context-distilled student from {save_dir}")
    student, _ = models.load(str(save_dir), dtype=dtype, device=args.device)
    student.eval()
    print("[phase5] eval context-distilled student no-context")
    dist_result = eval_mmlu.evaluate(student, tok, "no_context", dataset)
    print(f"  distilled accuracy: {dist_result['accuracy']:.3f}  ({dist_result['n_correct']}/{dist_result['n_total']})")

    summary = {
        "context": args.context,
        "n_questions": len(dataset),
        "base":      {"accuracy": base_result["accuracy"], "n_correct": base_result["n_correct"], "n_predicted": base_result["n_predicted"]},
        "in_context": {"accuracy": in_ctx_result["accuracy"], "n_correct": in_ctx_result["n_correct"], "n_predicted": in_ctx_result["n_predicted"]},
        "distilled": {"accuracy": dist_result["accuracy"], "n_correct": dist_result["n_correct"], "n_predicted": dist_result["n_predicted"]},
        "rows_base": base_result["rows"][:50],   # truncate to keep file size reasonable
        "rows_in_context": in_ctx_result["rows"][:50],
        "rows_distilled": dist_result["rows"][:50],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[phase5] wrote {out_path}")
    print(f"  base       acc = {base_result['accuracy']:.3f}")
    print(f"  in_context acc = {in_ctx_result['accuracy']:.3f}  (delta vs base: {in_ctx_result['accuracy']-base_result['accuracy']:+.3f})")
    print(f"  distilled  acc = {dist_result['accuracy']:.3f}  (delta vs base: {dist_result['accuracy']-base_result['accuracy']:+.3f})")


if __name__ == "__main__":
    main()
