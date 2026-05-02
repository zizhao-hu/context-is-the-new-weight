"""Experiment 01 — Context → Weight distillation.

Pipeline:
  1. Pick context C from the library (haiku, pirate, concise, ...).
  2. For N_train queries: teacher rollout M(Q | C), keep top-k logits per
     generated position.
  3. Distill student M → M_C on these targets, conditioning on the no-context
     prompt only. Default: LoRA r=16 on q,k,v,o projections.
  4. Eval on N_eval held-out queries: forward KL between M_teacher(Q|C) and
     M_student(Q|∅) along the teacher's greedy path; top-1 agreement;
     sampled output strings.
  5. Save the LoRA delta and an analysis JSON (per-module Frob norms, LoRA
     singular-value spectrum) for downstream experiments 02 and 03.

Usage (from repo root):
  python -m experiments.01_context_to_weight_distill.run \
    --config experiments/01_context_to_weight_distill/config.yaml \
    --context haiku
"""
from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import contexts as ctx_lib
from src import delta as delta_mod
from src import metrics
from src import models
from src import teacher as teacher_mod
from src.distill import (
    DistillDataset,
    build_examples,
    collate,
    make_lora,
    train_step,
)


def load_queries(path: Path, n: int, seed: int) -> list[str]:
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(items)
    return [it["query"] for it in items[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True, help="Name of the context from src/contexts.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.context not in ctx_lib.CONTEXTS:
        raise SystemExit(f"Unknown context: {args.context}. Known: {list(ctx_lib.CONTEXTS)}")

    out_dir = Path(cfg["out_dir"]) / args.context
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    queries_all = load_queries(
        Path(cfg["queries_path"]),
        cfg["n_train_queries"] + cfg["n_eval_queries"],
        cfg["seed"],
    )
    train_queries = queries_all[: cfg["n_train_queries"]]
    eval_queries = queries_all[cfg["n_train_queries"] :]

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    print(f"[load] {cfg['model']} on {args.device} dtype={dtype}")
    model, tok = models.load(cfg["model"], dtype=dtype, device=args.device)

    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"[teacher] rolling out N={len(train_queries)} queries with context={args.context}")
    traces = []
    for q in tqdm(train_queries):
        traces.append(
            teacher_mod.rollout(
                model, tok, args.context, q,
                max_new_tokens=cfg["teacher"]["max_new_tokens"],
                top_k=cfg["teacher"]["top_k"],
                temperature=cfg["teacher"]["temperature"],
            )
        )

    print("[student] preparing distillation set")
    examples = build_examples(traces)
    ds = DistillDataset(examples)
    dl = DataLoader(
        ds,
        batch_size=cfg["distill"]["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id=tok.pad_token_id),
    )

    if cfg["distill"]["method"] == "lora":
        print(f"[student] wrapping with LoRA r={cfg['distill']['lora']['r']}")
        model = make_lora(
            model,
            r=cfg["distill"]["lora"]["r"],
            alpha=cfg["distill"]["lora"]["alpha"],
            target_modules=cfg["distill"]["lora"]["target_modules"],
        )
        # PEFT freezes base params and only trains LoRA matrices.
    else:
        print("[student] full fine-tune")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["distill"]["lr"],
    )

    model.train()
    for epoch in range(cfg["distill"]["epochs"]):
        losses = []
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = train_step(model, batch, optim)
            losses.append(loss)
            pbar.set_postfix(loss=f"{loss:.4f}")
        print(f"[epoch {epoch}] mean KL = {sum(losses)/len(losses):.4f}")

    print("[eval] computing teacher/student divergence on held-out queries")
    model.eval()
    eval_rows = []
    for q in tqdm(eval_queries):
        # Teacher = original model with context. With LoRA, we can disable adapters
        # to get the teacher (= base) forward; otherwise we need a saved copy.
        if cfg["distill"]["method"] == "lora":
            with model.disable_adapter():
                t_logits, _ = metrics.teacher_student_logits(
                    model, model, tok, args.context, q,
                    max_new_tokens=cfg["teacher"]["max_new_tokens"],
                )
            # student = adapters enabled, no context
            with torch.no_grad():
                s_in = tok.apply_chat_template(
                    ctx_lib.build_messages("no_context", q),
                    return_tensors="pt", add_generation_prompt=True,
                ).to(args.device)
                s_logits = []
                t_argmax = t_logits.argmax(-1)
                for j in range(t_argmax.shape[0]):
                    out = model(s_in, use_cache=False).logits[:, -1, :]
                    s_logits.append(out[0].cpu())
                    s_in = torch.cat([s_in, t_argmax[j : j + 1].view(1, 1).to(args.device)], dim=-1)
                s_logits = torch.stack(s_logits)
        else:
            # need a separate teacher copy; for the full-FT path, save it earlier.
            raise NotImplementedError("Full-FT eval path not wired in this scaffold; use --method lora.")

        kl = metrics.per_token_kl(t_logits, s_logits).mean().item()
        top1 = metrics.top1_agreement(t_logits, s_logits)

        sample = ""
        if cfg["eval"].get("also_sample"):
            with torch.no_grad():
                ids = tok.apply_chat_template(
                    ctx_lib.build_messages("no_context", q),
                    return_tensors="pt", add_generation_prompt=True,
                ).to(args.device)
                gen = model.generate(ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
                sample = tok.decode(gen[0, ids.shape[1] :], skip_special_tokens=True)

        eval_rows.append({"query": q, "kl_mean": kl, "top1": top1, "student_sample": sample})

    (out_dir / "eval.json").write_text(json.dumps(eval_rows, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_dir / 'eval.json'}")

    print("[delta] computing ΔW analysis")
    student_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if cfg["distill"]["method"] == "lora":
        analysis = {
            "lora_spectrum": delta_mod.lora_rank_spectrum(student_state),
        }
    else:
        analysis = {
            "per_module": delta_mod.per_module_delta(base_state, student_state),
        }
    (out_dir / "delta_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    # Save adapter / full state. PEFT models have .save_pretrained.
    if cfg["distill"]["method"] == "lora":
        model.save_pretrained(out_dir / "lora")
    else:
        torch.save(student_state, out_dir / "student.pt")

    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
