"""Setting 3 — fine-tune on the context string itself.

For each context C, run standard LM fine-tuning on just the chat-template
encoding of the system prompt (+ any few-shot demos), with no synthetic
(Q, A) pairs. Saves the trained model under saves/ctxonly_<context>/ so
phase4 / phase4b can pick it up by overriding --context to ctxonly_<name>.

Usage (from repo root):
  python experiments/01_synth_distill_kvdw/phase2c_ctxonly.py \
    --config experiments/01_synth_distill_kvdw/config.yaml \
    --context haiku --n-steps 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src import distill, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # Save under a separate subdir so it doesn't clobber the synth-FT model.
    save_dir = Path(cfg["saves_root"]) / f"ctxonly_{args.context}"
    out_dir = Path(cfg["out_root"]) / f"ctxonly_{args.context}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase2c] context={args.context}  save_dir={save_dir}  n_steps={args.n_steps}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model, tok = models.load(cfg["model"], dtype=dtype, device=args.device)

    train_log = []
    for log in distill.train_context_only(
        model, tok, args.context,
        n_steps=args.n_steps,
        lr=cfg["phase2"]["lr"],
        grad_accum=cfg["phase2"]["grad_accum"],
        use_8bit_optim=cfg["phase2"]["use_8bit_optim"],
        enable_grad_ckpt=cfg["phase2"]["enable_grad_ckpt"],
        device=args.device,
    ):
        train_log.append(log)
        print(f"[phase2c] step={log['step']} loss={log['loss']:.4f}  (n_tokens={log['n_tokens']})")

    distill.save_full_ft(model, tok, save_dir)
    print(f"[phase2c] saved to {save_dir}")
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
