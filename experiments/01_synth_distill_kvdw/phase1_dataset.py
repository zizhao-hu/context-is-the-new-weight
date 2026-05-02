"""Phase 1 — generate the synthetic (Q, A) dataset for one context.

Usage:
  python experiments/01_synth_distill_kvdw/phase1_dataset.py \
    --config experiments/01_synth_distill_kvdw/config.yaml \
    --context haiku
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src import models, synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    n_queries = (
        cfg["phase1"]["n_queries_factual"]
        if args.context == "factual"
        else cfg["phase1"]["n_queries_default"]
    )

    out_path = Path(cfg["data_root"]) / f"{args.context}.jsonl"

    print(f"[phase1] context={args.context}  n_queries={n_queries}  out={out_path}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model, tok = models.load(cfg["model"], dtype=dtype, device=args.device)
    model.eval()

    written = synth.generate(
        model, tok, args.context, n_queries, out_path,
        max_new_tokens=cfg["phase1"]["max_new_tokens"],
        top_k=cfg["phase1"]["top_k"],
        temperature=cfg["phase1"]["temperature"],
        seed=cfg["seed"],
    )
    print(f"[phase1] wrote {written} curated rows to {out_path}")


if __name__ == "__main__":
    main()
