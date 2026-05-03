"""Aggregate MMLU-Pro results across all contexts into a single table.

Reads outputs/01_synth_distill_kvdw/<context>/mmlu_pro_eval.json for every
context that has it and prints a table comparing base / in_context /
distilled accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CONTEXTS = ["haiku", "pirate", "concise", "fewshot_translate_fr", "factual",
            "conv_history", "compressed_history"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--csv-path", default="figures/mmlu_pro_summary.csv")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    rows = []
    print(f"{'context':<22} {'base':>7} {'in_ctx':>7} {'distill':>7} {'dist-base':>10} {'in_ctx-base':>11} {'dist-in_ctx':>11}")
    print("-" * 80)
    for ctx in CONTEXTS:
        path = out_root / ctx / "mmlu_pro_eval.json"
        if not path.exists():
            print(f"{ctx:<22} (no mmlu_pro_eval.json)")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        b = d["base"]["accuracy"]
        ic = d["in_context"]["accuracy"]
        di = d["distilled"]["accuracy"]
        rows.append({
            "context": ctx,
            "base_acc": b,
            "in_context_acc": ic,
            "distilled_acc": di,
            "dist_minus_base": di - b,
            "in_context_minus_base": ic - b,
            "dist_minus_in_context": di - ic,
            "n_questions": d["n_questions"],
        })
        print(f"{ctx:<22} {b:>7.3f} {ic:>7.3f} {di:>7.3f} {di-b:>+10.3f} {ic-b:>+11.3f} {di-ic:>+11.3f}")

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
