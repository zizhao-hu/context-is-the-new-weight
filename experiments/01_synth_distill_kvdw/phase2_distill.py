"""Phase 2 — full fine-tune distillation on the synthetic dataset.

Usage:
  python experiments/01_synth_distill_kvdw/phase2_distill.py \
    --config experiments/01_synth_distill_kvdw/config.yaml \
    --context haiku
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src import distill, models, synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override config's phase2.epochs (use to retrain weak contexts)")
    ap.add_argument("--lr", type=float, default=None,
                    help="Override config's phase2.lr")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        cfg["phase2"]["epochs"] = args.epochs
        print(f"[phase2] epoch override: {args.epochs}")
    if args.lr is not None:
        cfg["phase2"]["lr"] = args.lr
        print(f"[phase2] lr override: {args.lr}")

    data_path = Path(cfg["data_root"]) / f"{args.context}.jsonl"
    save_dir = Path(cfg["saves_root"]) / args.context
    out_dir = Path(cfg["out_root"]) / args.context
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase2] context={args.context}  data={data_path}  save_dir={save_dir}")

    records = synth.load_dataset(data_path)
    print(f"[phase2] loaded {len(records)} training records")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model, tok = models.load(cfg["model"], dtype=dtype, device=args.device)

    train_log = []
    for log in distill.train_full_ft(
        model, tok, records,
        lr=cfg["phase2"]["lr"],
        epochs=cfg["phase2"]["epochs"],
        batch_size=cfg["phase2"]["batch_size"],
        grad_accum=cfg["phase2"]["grad_accum"],
        use_8bit_optim=cfg["phase2"]["use_8bit_optim"],
        enable_grad_ckpt=cfg["phase2"]["enable_grad_ckpt"],
        device=args.device,
    ):
        train_log.append(log)
        if len(train_log) % 5 == 0:
            print(f"[phase2] epoch={log['epoch']} step={log['step']} loss={log['loss']:.4f}")

    distill.save_full_ft(model, tok, save_dir)
    print(f"[phase2] saved trained model to {save_dir}")

    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")
    print(f"[phase2] wrote train log to {out_dir / 'train_log.json'}")


if __name__ == "__main__":
    main()
