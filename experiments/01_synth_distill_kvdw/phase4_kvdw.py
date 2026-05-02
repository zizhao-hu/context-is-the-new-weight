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
            v_ctx = kvdw.context_delta(base, with_ids, with_mask, no_ids, no_mask)
            v_dw = kvdw.weight_delta(base, trained, no_ids, no_mask)
            row = kvdw.per_site_alignment(v_ctx, v_dw, w_u=w_u)

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
