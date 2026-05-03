"""Phase 4e — per-layer activation deltas on training set vs validation set.

For one context, capture residual-stream activations at every post-attention
and post-MLP site for three runs:
  base on Q without context  -> A_base
  base on [C; Q]              -> A_ctx
  student (theta_C) on Q       -> A_stu

Compute per-layer mean L2 norms (averaged across attn + mlp sublayer sites
at each layer):
  ||v_ctx||      = ||A_ctx - A_base||      (in-context perturbation vs base)
  ||v_dw||       = ||A_stu - A_base||      (context-distillation-FT perturbation vs base)
  ||v_ctx - v_dw|| = ||A_ctx - A_stu||     (disagreement between S1 and S2)

Run on TWO query sets disjoint from each other:
  - TRAIN: 30 random queries actually used during the student's distillation
           (sampled from data/synth/<context>.jsonl).
  - VAL:   30 held-out queries from src/validation_queries.py.

Saves:
  outputs/01_synth_distill_kvdw/<context>/activations_train.json
  outputs/01_synth_distill_kvdw/<context>/activations_val.json

Usage:
  python experiments/01_synth_distill_kvdw/phase4e_activations_train_val.py \
    --config experiments/01_synth_distill_kvdw/config.yaml \
    --context haiku --n-queries 30
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from src import contexts as ctx_lib
from src import kvdw, models, synth, use_cases, validation_queries


def _chat_ids(tok, messages):
    out = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if isinstance(out, torch.Tensor):
        return out
    return out["input_ids"]


def _build_pair(tok, ctx_name: str, query: str, device):
    msgs_ctx = ctx_lib.build_messages(ctx_name, query)
    msgs_no = ctx_lib.build_messages("no_context", query)
    with_ids = _chat_ids(tok, msgs_ctx).to(device)
    no_ids = _chat_ids(tok, msgs_no).to(device)
    with_mask = torch.ones_like(with_ids)
    no_mask = torch.ones_like(no_ids)
    return (with_ids, with_mask), (no_ids, no_mask)


@torch.no_grad()
def compute_per_layer_norms(base, student, tok, ctx_name: str, queries: list[str], device):
    """For each query, capture residuals under three runs and compute per-site
    L2 norms of the three perturbation vectors. Aggregate to per-layer means
    (averaging across attn + mlp at each layer, then across queries)."""
    accum: dict[int, dict[str, list[float]]] = {}
    n_ok = 0
    for q in queries:
        try:
            (with_ids, with_mask), (no_ids, no_mask) = _build_pair(tok, ctx_name, q, device)
            cap_base = kvdw.captures_for(base, no_ids, no_mask)
            cap_ctx = kvdw.captures_for(base, with_ids, with_mask)
            cap_stu = kvdw.captures_for(student, no_ids, no_mask)

            # Each capture is dict[(layer, "attn"|"mlp")] -> (B, hidden)
            for key in cap_base:
                layer, _site = key
                a = cap_base[key].float()
                b = cap_ctx.get(key, None)
                c = cap_stu.get(key, None)
                if b is None or c is None:
                    continue
                v_ctx = (b.float() - a)
                v_dw = (c.float() - a)
                diff = v_ctx - v_dw

                ctx_n = v_ctx.norm(dim=-1).mean().item()
                dw_n = v_dw.norm(dim=-1).mean().item()
                diff_n = diff.norm(dim=-1).mean().item()
                base_n = a.norm(dim=-1).mean().item()

                d = accum.setdefault(layer, {"ctx": [], "dw": [], "diff": [], "base": []})
                d["ctx"].append(ctx_n)
                d["dw"].append(dw_n)
                d["diff"].append(diff_n)
                d["base"].append(base_n)
            n_ok += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

    per_layer = []
    for layer in sorted(accum):
        d = accum[layer]
        per_layer.append({
            "layer": layer,
            "v_ctx_norm": sum(d["ctx"]) / max(1, len(d["ctx"])),
            "v_dw_norm": sum(d["dw"]) / max(1, len(d["dw"])),
            "v_ctx_minus_v_dw_norm": sum(d["diff"]) / max(1, len(d["diff"])),
            "base_norm": sum(d["base"]) / max(1, len(d["base"])),
        })
    return {"n_queries": n_ok, "per_layer": per_layer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--n-queries", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    save_dir = Path(cfg["saves_root"]) / args.context
    out_dir = Path(cfg["out_root"]) / args.context
    out_dir.mkdir(parents=True, exist_ok=True)

    base_context = args.context[len("ctxonly_"):] if args.context.startswith("ctxonly_") else args.context

    # TRAIN: random sample from the training synth dataset
    data_path = Path(cfg["data_root"]) / f"{base_context}.jsonl"
    records = synth.load_dataset(data_path)
    rng = random.Random(cfg["seed"] + 12345)
    train_queries = [r["query"] for r in rng.sample(records, min(args.n_queries, len(records)))]

    # VAL: held-out validation queries
    if base_context == "factual":
        val_queries = validation_queries.sample_validation(
            total=args.n_queries, recall_facts=use_cases.FACTUAL_KEYS,
        )
    else:
        val_queries = validation_queries.sample_validation(total=args.n_queries)

    print(f"[phase4e] context={args.context} (base={base_context})")
    print(f"[phase4e] n_train={len(train_queries)} n_val={len(val_queries)}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base, tok = models.load(cfg["model"], dtype=dtype, device=args.device)
    student, _ = models.load(str(save_dir), dtype=dtype, device=args.device)
    base.eval()
    student.eval()

    print("[phase4e] computing TRAIN activations")
    train_result = compute_per_layer_norms(base, student, tok, base_context, train_queries, args.device)
    train_result["context"] = args.context
    train_result["query_set"] = "train"
    (out_dir / "activations_train.json").write_text(json.dumps(train_result, indent=2), encoding="utf-8")
    print(f"[phase4e] wrote activations_train.json ({train_result['n_queries']} queries)")

    print("[phase4e] computing VAL activations")
    val_result = compute_per_layer_norms(base, student, tok, base_context, val_queries, args.device)
    val_result["context"] = args.context
    val_result["query_set"] = "val"
    (out_dir / "activations_val.json").write_text(json.dumps(val_result, indent=2), encoding="utf-8")
    print(f"[phase4e] wrote activations_val.json ({val_result['n_queries']} queries)")


if __name__ == "__main__":
    main()
