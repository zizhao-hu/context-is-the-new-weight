"""Phase 5x — MMLU-Pro accuracy for any PEFT-adapter model.

Loads a PEFT adapter (lora / prompt / prefix) on top of the base model and
evaluates it on MMLU-Pro. Saves to
outputs/01_synth_distill_kvdw/<method>_<context>/mmlu_pro_eval.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src import eval_mmlu, models, peft_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--method", required=True, choices=["lora", "prompt", "prefix"])
    ap.add_argument("--eval-path", default="data/mmlu_pro_eval.jsonl")
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_context = args.context

    adapter_dir = Path(cfg["saves_root"]) / f"{args.method}_{base_context}"
    out_dir = Path(cfg["out_root"]) / f"{args.method}_{base_context}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mmlu_pro_eval.json"

    eval_path = Path(args.eval_path)
    dataset = eval_mmlu.load_jsonl(eval_path)[: args.n_questions]
    print(f"[phase5x:{args.method}] context={args.context}  n_questions={len(dataset)}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base, tok = models.load(cfg["model"], dtype=dtype, device=args.device)
    adapter_model = peft_adapter.load_adapter(base, adapter_dir)
    adapter_model.eval()

    print(f"[phase5x:{args.method}] eval adapter no-context")
    result = eval_mmlu.evaluate(adapter_model, tok, "no_context", dataset)
    print(f"  {args.method} accuracy: {result['accuracy']:.3f}  ({result['n_correct']}/{result['n_total']})")

    summary = {
        "context": args.context,
        "method": args.method,
        "n_questions": len(dataset),
        "adapter": {"accuracy": result["accuracy"], "n_correct": result["n_correct"], "n_predicted": result["n_predicted"]},
        "rows_adapter": result["rows"][:50],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[phase5x:{args.method}] wrote {out_path}")


if __name__ == "__main__":
    main()
