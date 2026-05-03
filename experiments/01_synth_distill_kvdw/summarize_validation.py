"""Combine ROUGE + behavioral-pattern validation across all contexts.

Reads each context's `rouge_validation.json`, recomputes per-context
behavioural pattern rates (haiku-shape, pirate-word, refusal, ...), and
prints one table with both ROUGE gap-closure and the behavioural deltas.

Usage:
  python experiments/01_synth_distill_kvdw/summarize_validation.py \
    --out-root /scratch1/zizhaoh/context-is-the-new-weight/outputs/01_synth_distill_kvdw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import behavior_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    contexts = ["haiku", "pirate", "concise", "fewshot_translate_fr", "factual"]

    print(f"{'context':<22} {'R(b,c)':>7} {'R(s,c)':>7} {'gap_L':>6}  | {'pattern (teacher / student / base)':<42}")
    print("-" * 100)

    for c in contexts:
        path = out_root / c / "rouge_validation.json"
        if not path.exists():
            print(f"{c:<22}  no rouge_validation.json")
            continue
        d = json.load(path.open(encoding="utf-8"))
        agg = d["aggregate"]
        rows = d["rows"]

        teacher = [r["a_ctx"] for r in rows]
        student = [r["a_student"] for r in rows]
        base = [r["a_base"] for r in rows]

        bt = behavior_eval.score_behavior(teacher, c)
        bs = behavior_eval.score_behavior(student, c)
        bb = behavior_eval.score_behavior(base, c)

        # Pick the most informative pattern metric per context
        if c == "haiku":
            tag = "haiku_shaped"
            t, s, b = bt["haiku_shaped_rate"], bs["haiku_shaped_rate"], bb["haiku_shaped_rate"]
        elif c == "pirate":
            tag = "pirate_word"
            t, s, b = bt["pirate_word_rate"], bs["pirate_word_rate"], bb["pirate_word_rate"]
        elif c == "concise":
            tag = "short_answer"
            t, s, b = bt["short_answer_rate"], bs["short_answer_rate"], bb["short_answer_rate"]
        elif c == "fewshot_translate_fr":
            tag = "french_rate"
            t, s, b = bt["french_rate"], bs["french_rate"], bb["french_rate"]
        elif c == "factual":
            tag = "refusal"
            t, s, b = bt["refusal_rate"], bs["refusal_rate"], bb["refusal_rate"]
        else:
            tag, t, s, b = "?", 0.0, 0.0, 0.0

        bc = agg["rougeL_base_ctx_mean"]
        sc = agg["rougeL_stu_ctx_mean"]
        gc = agg["gap_closure_rougeL"]

        # Behavioural gap closure: how much of the base->teacher gap in pattern rate did the student close?
        denom = (t - b) if abs(t - b) > 1e-6 else 1.0
        beh_close = (s - b) / denom

        pat_str = f"{tag}={t:.2f}/{s:.2f}/{b:.2f}  beh_close={beh_close:+.2f}"
        print(f"{c:<22} {bc:>7.3f} {sc:>7.3f} {gc:>6.3f}  | {pat_str}")

    print()
    print("Legend: R(b,c)=ROUGE-L(base, ctx); R(s,c)=ROUGE-L(student, ctx); gap_L=(R(s,c)-R(b,c))/(1-R(b,c))")
    print("        pattern triplet = teacher / student / base rate of the pattern (haiku-shape, pirate-word, etc.)")
    print("        beh_close = (student_rate - base_rate) / (teacher_rate - base_rate)")


if __name__ == "__main__":
    main()
