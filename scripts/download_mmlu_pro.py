"""Download MMLU-Pro and save a stratified subset for eval and OOD probes.

Run on a CARC login node (has internet).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=300, help="Total eval questions")
    ap.add_argument("--n-ood-probes", type=int, default=30, help="Subset for activation OOD probe")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    print("[download] loading TIGER-Lab/MMLU-Pro test split")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"[download] {len(ds)} total questions across {len(set(ds['category']))} categories")

    # Stratified sample by category for the eval set
    by_cat: dict = defaultdict(list)
    for r in ds:
        by_cat[r["category"]].append(r)
    cats = sorted(by_cat)
    rng = random.Random(args.seed)

    per_cat = max(1, args.n_eval // len(cats))
    eval_records = []
    for c in cats:
        items = by_cat[c]
        rng.shuffle(items)
        eval_records.extend(items[:per_cat])
    rng.shuffle(eval_records)
    eval_records = eval_records[: args.n_eval]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_path = out_dir / "mmlu_pro_eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as f:
        for r in eval_records:
            # keep only the fields we need to keep file size small
            keep = {
                "question_id": r.get("question_id"),
                "category": r.get("category"),
                "question": r.get("question"),
                "options": r.get("options"),
                "answer": r.get("answer"),
                "answer_index": r.get("answer_index"),
            }
            f.write(json.dumps(keep, ensure_ascii=False) + "\n")
    print(f"[download] wrote {len(eval_records)} eval questions -> {eval_path}")

    # OOD probe set: a smaller subset, only the question text (no need for options
    # for activation capture since we just need the model to attend to the question)
    ood_probes = rng.sample(eval_records, min(args.n_ood_probes, len(eval_records)))
    ood_path = out_dir / "ood_mmlu_pro.jsonl"
    with ood_path.open("w", encoding="utf-8") as f:
        for r in ood_probes:
            # Format as a plain query for activation probing
            opts = r.get("options", [])
            opt_text = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
            query = f"Question: {r['question']}\n\nOptions:\n{opt_text}\n\nAnswer:"
            f.write(json.dumps({"category": r.get("category"), "query": query}) + "\n")
    print(f"[download] wrote {len(ood_probes)} OOD probes -> {ood_path}")


if __name__ == "__main__":
    main()
