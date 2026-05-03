"""Phase 2-prefix — train a per-layer K/V prefix on the synth (Q, A) set.

For one context, train a small prefix (default 16 virtual tokens) by
distillation on the same (Q, A) pairs used for context-distillation.
The prefix is the only trainable parameter set; base model is frozen.

Saves the PEFT adapter (KB-MB scale) at:
  saves/prefix_<context>/

Usage:
  python experiments/01_synth_distill_kvdw/phase2p_prefix.py \
    --config experiments/01_synth_distill_kvdw/config.yaml \
    --context haiku --num-virtual-tokens 16 --epochs 3 --lr 5e-3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src import models, prefix_tune, synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--num-virtual-tokens", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_context = args.context
    save_dir = Path(cfg["saves_root"]) / f"prefix_{base_context}"
    out_dir = Path(cfg["out_root"]) / f"prefix_{base_context}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg["data_root"]) / f"{base_context}.jsonl"
    records = synth.load_dataset(data_path)
    print(f"[phase2p] context={args.context}  n_records={len(records)}  num_virtual_tokens={args.num_virtual_tokens}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base, tok = models.load(cfg["model"], dtype=dtype, device=args.device)
    model = prefix_tune.make_prefix_model(base, num_virtual_tokens=args.num_virtual_tokens)

    train_log = []
    for log in prefix_tune.train_prefix(
        model, tok, records,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=cfg["phase2"]["batch_size"],
        grad_accum=cfg["phase2"]["grad_accum"],
        use_8bit_optim=cfg["phase2"]["use_8bit_optim"],
        device=args.device,
    ):
        train_log.append(log)
        if len(train_log) % 5 == 0:
            print(f"  step={log['step']} loss={log['loss']:.4f}")

    prefix_tune.save_prefix(model, save_dir)
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")
    print(f"[phase2p] saved prefix adapter to {save_dir}")
    print(f"[phase2p] {train_log[-1]['n_trainable']:,} trainable parameters")


if __name__ == "__main__":
    main()
