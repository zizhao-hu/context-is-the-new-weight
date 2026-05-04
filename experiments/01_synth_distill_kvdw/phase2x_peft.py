"""Phase 2x — train any PEFT adapter (lora / prompt / prefix) on the synth (Q, A)
pairs for one context. Saves the adapter at saves/<method>_<context>/.

Usage:
  python experiments/01_synth_distill_kvdw/phase2x_peft.py \
    --config experiments/01_synth_distill_kvdw/config.yaml \
    --context haiku --method lora

The lr default is method-specific (LoRA: 3e-4; prompt/prefix: 1e-2). Override
via --lr.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src import models, peft_adapter, synth


METHOD_DEFAULTS = {
    "lora":   {"lr": 3e-4, "epochs": 3},
    "prompt": {"lr": 1e-2, "epochs": 6},   # prompt tuning is the slowest to converge
    "prefix": {"lr": 1e-2, "epochs": 3},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--method", required=True, choices=["lora", "prompt", "prefix"])
    ap.add_argument("--num-virtual-tokens", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    defaults = METHOD_DEFAULTS[args.method]
    lr = args.lr if args.lr is not None else defaults["lr"]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_context = args.context
    save_dir = Path(cfg["saves_root"]) / f"{args.method}_{base_context}"
    out_dir = Path(cfg["out_root"]) / f"{args.method}_{base_context}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg["data_root"]) / f"{base_context}.jsonl"
    records = synth.load_dataset(data_path)
    print(f"[phase2x:{args.method}] context={args.context}  n_records={len(records)}  "
          f"epochs={epochs}  lr={lr}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base, tok = models.load(cfg["model"], dtype=dtype, device=args.device)
    model = peft_adapter.make_adapter(
        base,
        method=args.method,
        num_virtual_tokens=args.num_virtual_tokens,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    train_log = []
    for log in peft_adapter.train_adapter(
        model, tok, records,
        lr=lr,
        epochs=epochs,
        batch_size=cfg["phase2"]["batch_size"],
        grad_accum=cfg["phase2"]["grad_accum"],
        use_8bit_optim=cfg["phase2"]["use_8bit_optim"],
        device=args.device,
    ):
        train_log.append(log)
        if len(train_log) % 5 == 0:
            print(f"  step={log['step']} loss={log['loss']:.4f}")

    peft_adapter.save_adapter(model, save_dir)
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")
    print(f"[phase2x:{args.method}] saved adapter to {save_dir}")
    if train_log:
        print(f"[phase2x:{args.method}] {train_log[-1]['n_trainable']:,} trainable parameters")


if __name__ == "__main__":
    main()
