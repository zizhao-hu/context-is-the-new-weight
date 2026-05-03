"""Phase 5p — MMLU-Pro accuracy for prefix-tuned model.

Same as phase5_mmlu_eval but loads a PEFT prefix adapter on top of the
base model and evaluates that as the third variant. Saves to
outputs/01_synth_distill_kvdw/prefix_<context>/mmlu_pro_eval.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src import eval_mmlu, models, prefix_tune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--eval-path", default="data/mmlu_pro_eval.jsonl")
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_context = args.context

    prefix_dir = Path(cfg["saves_root"]) / f"prefix_{base_context}"
    out_dir = Path(cfg["out_root"]) / f"prefix_{base_context}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mmlu_pro_eval.json"

    eval_path = Path(args.eval_path)
    dataset = eval_mmlu.load_jsonl(eval_path)[: args.n_questions]
    print(f"[phase5p] context={args.context}  n_questions={len(dataset)}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base, tok = models.load(cfg["model"], dtype=dtype, device=args.device)
    prefix_model = prefix_tune.load_prefix(base, prefix_dir)
    prefix_model.eval()

    print("[phase5p] eval prefix-tuned model no-context")
    pref_result = eval_mmlu.evaluate(prefix_model, tok, "no_context", dataset)
    print(f"  prefix accuracy: {pref_result['accuracy']:.3f}  ({pref_result['n_correct']}/{pref_result['n_total']})")

    summary = {
        "context": args.context,
        "n_questions": len(dataset),
        "prefix": {"accuracy": pref_result["accuracy"], "n_correct": pref_result["n_correct"], "n_predicted": pref_result["n_predicted"]},
        "rows_prefix": pref_result["rows"][:50],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[phase5p] wrote {out_path}")


if __name__ == "__main__":
    main()
