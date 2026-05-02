"""Phase 4 — KV-vs-ΔW residual-stream comparison.

For ~30 held-out probe queries, capture the residual stream at every
post-attention and post-MLP site, and compute v_ctx vs v_dw alignment per site.

Output: outputs/01_synth_distill_kvdw/<context>/kvdw.json with per-site cosine
metrics aggregated across the probe set.

Usage:
  python experiments/01_synth_distill_kvdw/phase4_kvdw.py \
    --config experiments/01_synth_distill_kvdw/config.yaml \
    --context haiku
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from src import contexts as ctx_lib
from src import kvdw, models, use_cases


def _chat_ids(tok, messages):
    out = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if isinstance(out, torch.Tensor):
        return out
    return out["input_ids"]


def _build_pair(tok, ctx_name: str, query: str, device):
    """Return ((with_ctx_ids, with_ctx_mask), (no_ctx_ids, no_ctx_mask)) on device."""
    msgs_ctx = ctx_lib.build_messages(ctx_name, query)
    msgs_noctx = ctx_lib.build_messages("no_context", query)
    with_ids = _chat_ids(tok, msgs_ctx).to(device)
    no_ids = _chat_ids(tok, msgs_noctx).to(device)
    with_mask = torch.ones_like(with_ids)
    no_mask = torch.ones_like(no_ids)
    return (with_ids, with_mask), (no_ids, no_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    save_dir = Path(cfg["saves_root"]) / args.context
    out_dir = Path(cfg["out_root"]) / args.context
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kvdw.json"

    print(f"[phase4] context={args.context}  trained={save_dir}")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base, tok = models.load(cfg["model"], dtype=dtype, device=args.device)
    trained, _ = models.load(str(save_dir), dtype=dtype, device=args.device)
    base.eval()
    trained.eval()

    # Build a fresh held-out probe set by sampling queries with a different seed
    # than Phase 1 used. Use a generic query mix (no recall) so probes work
    # across all contexts.
    rng = random.Random(cfg["seed"] + 9999)
    probe_queries = use_cases.sample_queries(
        seed=cfg["seed"] + 9999,
        total=cfg["phase4"]["n_probe_queries"],
    )

    # Get unembedding for tokenspace cosine. HF Llama: lm_head.weight (V, hidden).
    w_u = base.lm_head.weight.detach()

    # Aggregate per-site sums to compute mean across probes.
    site_keys = None
    accum: dict = {}
    n_probes = 0

    for q in probe_queries:
        try:
            (with_ids, with_mask), (no_ids, no_mask) = _build_pair(tok, args.context, q, args.device)
            # Three triangle vertices per site: base no-ctx, base+ctx, trained no-ctx.
            cap_base = kvdw.captures_for(base, no_ids, no_mask)
            cap_with = kvdw.captures_for(base, with_ids, with_mask)
            cap_trained = kvdw.captures_for(trained, no_ids, no_mask)
            v_ctx = {k: cap_with[k] - cap_base[k] for k in cap_with.keys() & cap_base.keys()}
            v_dw = {k: cap_trained[k] - cap_base[k] for k in cap_trained.keys() & cap_base.keys()}
            row = kvdw.per_site_alignment(v_ctx, v_dw, w_u=w_u)

            # Augment per-site row with triangle metrics:
            #   cos(v_ctx, base), cos(v_dw, base): direction of perturbation vs base flow.
            #   base_norm: scale of the base residual at that site.
            #   v_ctx_to_v_dw_dist: ‖v_ctx − v_dw‖ — third side length.
            for k in list(row.keys()):
                # row keys are like "L26_attn"; reconstruct original tuple key
                layer = int(k[1:3])
                site = k[4:]
                tk = (layer, site)
                if tk not in cap_base:
                    continue
                a_vec = cap_base[tk]
                b_vec = v_ctx[tk]
                c_vec = v_dw[tk]
                base_norm = a_vec.float().norm(dim=-1).mean().item()
                cos_ctx_base = kvdw.cosine(b_vec, a_vec).mean().item()
                cos_dw_base = kvdw.cosine(c_vec, a_vec).mean().item()
                third_side = (b_vec.float() - c_vec.float()).norm(dim=-1).mean().item()
                row[k]["base_norm"] = base_norm
                row[k]["cos_ctx_base"] = cos_ctx_base
                row[k]["cos_dw_base"] = cos_dw_base
                row[k]["v_ctx_minus_v_dw_norm"] = third_side
                row[k]["ctx_relative_norm"] = row[k]["ctx_norm"] / max(1e-8, base_norm)
                row[k]["dw_relative_norm"] = row[k]["dw_norm"] / max(1e-8, base_norm)

            if site_keys is None:
                site_keys = sorted(row.keys())
                for k in site_keys:
                    accum[k] = {m: 0.0 for m in row[k]}
            for k in site_keys:
                for m, v in row[k].items():
                    accum[k][m] += v
            n_probes += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[phase4] OOM on query (skipped): {q[:60]}...")
            continue

    if n_probes == 0:
        raise SystemExit("[phase4] no probes succeeded; aborting.")

    # Mean across probes
    per_site = {
        k: {m: accum[k][m] / n_probes for m in accum[k]}
        for k in site_keys
    }

    result = {
        "context": args.context,
        "n_probes": n_probes,
        "n_sites": len(per_site),
        "per_site": per_site,
    }

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[phase4] wrote {out_path}  ({n_probes} probes, {len(per_site)} sites)")


if __name__ == "__main__":
    main()
