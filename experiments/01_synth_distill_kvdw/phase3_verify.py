"""Phase 3 — behavior verification (the gating step).

Run the trained student on its own training queries with no context. Compare
against the teacher's stored answers. If exact-match / prefix-match rates are
below threshold, distillation underfit and Phase 4 should not run.

Usage:
  python experiments/01_synth_distill_kvdw/phase3_verify.py \
    --config experiments/01_synth_distill_kvdw/config.yaml \
    --context haiku
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src import models, synth, verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    data_path = Path(cfg["data_root"]) / f"{args.context}.jsonl"
    save_dir = Path(cfg["saves_root"]) / args.context
    out_dir = Path(cfg["out_root"]) / args.context
    out_path = out_dir / "behavior_verify.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = synth.load_dataset(data_path)
    print(f"[phase3] context={args.context}  loaded {len(records)} records  student={save_dir}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    student, tok = models.load(str(save_dir), dtype=dtype, device=args.device)
    base, _ = models.load(cfg["model"], dtype=dtype, device=args.device)
    student.eval()
    base.eval()

    agg = verify.run(
        student, base, tok, records, out_path,
        max_new_tokens=cfg["phase3"]["max_new_tokens"],
        sample_for_kl=cfg["phase3"]["sample_for_kl"],
    )

    print("[phase3] aggregate metrics:")
    print(json.dumps(agg, indent=2))
    print(f"[phase3] wrote {out_path}")

    # Hard gate signal — caller (run.sh) checks the file. Here we just print
    # a status line that's easy to grep.
    if agg["exact_match_rate"] >= 0.30 or agg["prefix_match_rate_ge_5"] >= 0.50:
        print("[phase3] STATUS=PASS")
    else:
        print("[phase3] STATUS=WEAK — eyeball samples before deciding to run phase4")


if __name__ == "__main__":
    main()
