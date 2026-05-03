"""Phase 4-prefix — compare prefix-tuned activations vs in-context vs base.

For one context with a trained prefix adapter, compute three perturbation
vectors at every post-attention and post-MLP residual-stream site:
  v_ctx    = resid([C; Q]; base)        - resid(Q; base)        # in-context
  v_dist   = resid(Q; theta_C)          - resid(Q; base)        # context-distilled (full FT)
  v_prefix = resid(Q; base + prefix)    - resid(Q; base)        # prefix-tuned

Then per layer aggregate the L2 norms and pairwise cosines between these
three vectors. The headline question: is v_prefix more similar to v_ctx
than v_dist is?

Saves outputs/01_synth_distill_kvdw/<context>/prefix_activations.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src import contexts as ctx_lib
from src import kvdw, models, prefix_tune, synth, use_cases, validation_queries


def _chat_ids(tok, messages):
    out = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if isinstance(out, torch.Tensor):
        return out
    return out["input_ids"]


def _build_pair(tok, ctx_name, query, device):
    msgs_ctx = ctx_lib.build_messages(ctx_name, query)
    msgs_no = ctx_lib.build_messages("no_context", query)
    with_ids = _chat_ids(tok, msgs_ctx).to(device)
    no_ids = _chat_ids(tok, msgs_no).to(device)
    return (with_ids, torch.ones_like(with_ids)), (no_ids, torch.ones_like(no_ids))


@torch.no_grad()
def compute_three_way(base, distilled, prefix_model, tok, ctx_name, queries, device):
    """For each query, compute v_ctx, v_dist, v_prefix per (layer, site)
    and aggregate to per-layer means. Also compute pairwise cosines."""
    accum: dict[int, dict] = {}
    n_ok = 0

    for q in queries:
        try:
            (with_ids, with_mask), (no_ids, no_mask) = _build_pair(tok, ctx_name, q, device)
            cap_base = kvdw.captures_for(base, no_ids, no_mask)
            cap_ctx = kvdw.captures_for(base, with_ids, with_mask)
            cap_dist = kvdw.captures_for(distilled, no_ids, no_mask)
            cap_prefix = kvdw.captures_for(prefix_model, no_ids, no_mask)

            for key in cap_base:
                if key not in cap_ctx or key not in cap_dist or key not in cap_prefix:
                    continue
                a = cap_base[key].float()
                v_ctx = (cap_ctx[key].float() - a)
                v_dist = (cap_dist[key].float() - a)
                v_prefix = (cap_prefix[key].float() - a)

                layer, _ = key
                d = accum.setdefault(layer, {
                    "v_ctx_n": [], "v_dist_n": [], "v_prefix_n": [],
                    "cos_ctx_dist": [], "cos_ctx_prefix": [], "cos_dist_prefix": [],
                    "ctx_minus_prefix_n": [], "ctx_minus_dist_n": [],
                })
                d["v_ctx_n"].append(v_ctx.norm(dim=-1).mean().item())
                d["v_dist_n"].append(v_dist.norm(dim=-1).mean().item())
                d["v_prefix_n"].append(v_prefix.norm(dim=-1).mean().item())
                d["cos_ctx_dist"].append(F.cosine_similarity(v_ctx, v_dist, dim=-1).mean().item())
                d["cos_ctx_prefix"].append(F.cosine_similarity(v_ctx, v_prefix, dim=-1).mean().item())
                d["cos_dist_prefix"].append(F.cosine_similarity(v_dist, v_prefix, dim=-1).mean().item())
                d["ctx_minus_prefix_n"].append((v_ctx - v_prefix).norm(dim=-1).mean().item())
                d["ctx_minus_dist_n"].append((v_ctx - v_dist).norm(dim=-1).mean().item())
            n_ok += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

    per_layer = []
    for layer in sorted(accum):
        d = accum[layer]
        per_layer.append({
            "layer": layer,
            "v_ctx_norm": sum(d["v_ctx_n"]) / max(1, len(d["v_ctx_n"])),
            "v_dist_norm": sum(d["v_dist_n"]) / max(1, len(d["v_dist_n"])),
            "v_prefix_norm": sum(d["v_prefix_n"]) / max(1, len(d["v_prefix_n"])),
            "cos_ctx_dist": sum(d["cos_ctx_dist"]) / max(1, len(d["cos_ctx_dist"])),
            "cos_ctx_prefix": sum(d["cos_ctx_prefix"]) / max(1, len(d["cos_ctx_prefix"])),
            "cos_dist_prefix": sum(d["cos_dist_prefix"]) / max(1, len(d["cos_dist_prefix"])),
            "ctx_minus_prefix_norm": sum(d["ctx_minus_prefix_n"]) / max(1, len(d["ctx_minus_prefix_n"])),
            "ctx_minus_dist_norm": sum(d["ctx_minus_dist_n"]) / max(1, len(d["ctx_minus_dist_n"])),
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
    base_context = args.context

    distilled_dir = Path(cfg["saves_root"]) / base_context
    prefix_dir = Path(cfg["saves_root"]) / f"prefix_{base_context}"
    out_dir = Path(cfg["out_root"]) / f"prefix_{base_context}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase4p] context={args.context}")
    print(f"[phase4p]   distilled at {distilled_dir}")
    print(f"[phase4p]   prefix at {prefix_dir}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base, tok = models.load(cfg["model"], dtype=dtype, device=args.device)
    distilled, _ = models.load(str(distilled_dir), dtype=dtype, device=args.device)
    base_for_prefix, _ = models.load(cfg["model"], dtype=dtype, device=args.device)
    prefix_model = prefix_tune.load_prefix(base_for_prefix, prefix_dir)
    base.eval()
    distilled.eval()
    prefix_model.eval()

    # Build query sets
    rng = random.Random(cfg["seed"] + 12345)
    data_path = Path(cfg["data_root"]) / f"{base_context}.jsonl"
    records = synth.load_dataset(data_path)
    train_queries = [r["query"] for r in rng.sample(records, min(args.n_queries, len(records)))]

    if base_context == "factual":
        val_queries = validation_queries.sample_validation(total=args.n_queries, recall_facts=use_cases.FACTUAL_KEYS)
    else:
        val_queries = validation_queries.sample_validation(total=args.n_queries)

    ood_path = Path("data/ood_mmlu_pro.jsonl")
    ood_queries: list[str] = []
    if ood_path.exists():
        for line in ood_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ood_queries.append(json.loads(line)["query"])
                if len(ood_queries) >= args.n_queries:
                    break

    out: dict = {"context": args.context}
    for split, qs in [("train", train_queries), ("val", val_queries), ("ood", ood_queries)]:
        if not qs:
            continue
        print(f"[phase4p] computing {split} ({len(qs)} queries)")
        out[split] = compute_three_way(base, distilled, prefix_model, tok, base_context, qs, args.device)

    (out_dir / "prefix_activations.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[phase4p] wrote {out_dir / 'prefix_activations.json'}")


if __name__ == "__main__":
    main()
