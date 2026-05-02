"""Setting 3 pipeline — train on context-only, then verify + KV-vs-ΔW + ΔW Frob.

This is Setting 3 of the three model-change schemas:
  Setting 1 (in-context):       base + C at inference (no training)
  Setting 2 (synth-FT):          fine-tune on (Q, A_with_C) pairs, no C at inference
  Setting 3 (ctxonly-FT, here):  fine-tune on the context string itself, no Q/A pairs

Outputs land under <out_root>/ctxonly_<context>/ and <saves_root>/ctxonly_<context>/.

Usage:
  python experiments/01_synth_distill_kvdw/run_ctxonly_pipeline.py \
    --config experiments/01_synth_distill_kvdw/config.yaml \
    --context haiku --n-steps 100
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM

from src import contexts as ctx_lib
from src import distill, kvdw, models, synth, use_cases, verify


def _chat_ids(tok, messages):
    out = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if isinstance(out, torch.Tensor):
        return out
    return out["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    save_dir = Path(cfg["saves_root"]) / f"ctxonly_{args.context}"
    out_dir = Path(cfg["out_root"]) / f"ctxonly_{args.context}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Phase 2c: train on context only
    # ============================================================
    print(f"\n===== Phase 2c (ctxonly) for {args.context} =====")
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
        if len(train_log) % 5 == 0:
            print(f"  step={log['step']} loss={log['loss']:.4f}")
    distill.save_full_ft(model, tok, save_dir)
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")
    print(f"  saved to {save_dir}")
    del model
    torch.cuda.empty_cache()

    # ============================================================
    # Phase 3: verify on the original context's synth dataset
    # ============================================================
    print(f"\n===== Phase 3 verify (against synth/{args.context}.jsonl) =====")
    data_path = Path(cfg["data_root"]) / f"{args.context}.jsonl"
    records = synth.load_dataset(data_path)
    student, tok = models.load(str(save_dir), dtype=dtype, device=args.device)
    base, _ = models.load(cfg["model"], dtype=dtype, device=args.device)
    student.eval()
    base.eval()
    agg = verify.run(
        student, base, tok, records, out_dir / "behavior_verify.json",
        max_new_tokens=cfg["phase3"]["max_new_tokens"],
        sample_for_kl=cfg["phase3"]["sample_for_kl"],
    )
    print("  aggregate:", json.dumps(agg, indent=2))

    # ============================================================
    # Phase 4: KV-vs-ΔW (v_ctx from base+C, v_dw from this student)
    # ============================================================
    print(f"\n===== Phase 4 KV-vs-ΔW =====")
    rng = random.Random(cfg["seed"] + 9999)
    probe_queries = use_cases.sample_queries(
        seed=cfg["seed"] + 9999, total=cfg["phase4"]["n_probe_queries"]
    )
    w_u = base.lm_head.weight.detach()

    site_keys = None
    accum = {}
    n_probes = 0
    for q in probe_queries:
        try:
            msgs_ctx = ctx_lib.build_messages(args.context, q)
            msgs_no = ctx_lib.build_messages("no_context", q)
            with_ids = _chat_ids(tok, msgs_ctx).to(args.device)
            no_ids = _chat_ids(tok, msgs_no).to(args.device)
            with_mask = torch.ones_like(with_ids)
            no_mask = torch.ones_like(no_ids)

            cap_base = kvdw.captures_for(base, no_ids, no_mask)
            cap_with = kvdw.captures_for(base, with_ids, with_mask)
            cap_trained = kvdw.captures_for(student, no_ids, no_mask)
            v_ctx = {k: cap_with[k] - cap_base[k] for k in cap_with.keys() & cap_base.keys()}
            v_dw = {k: cap_trained[k] - cap_base[k] for k in cap_trained.keys() & cap_base.keys()}
            row = kvdw.per_site_alignment(v_ctx, v_dw, w_u=w_u)

            for k in list(row.keys()):
                layer = int(k[1:3])
                site = k[4:]
                tk = (layer, site)
                if tk not in cap_base:
                    continue
                a_vec = cap_base[tk]
                b_vec = v_ctx[tk]
                c_vec = v_dw[tk]
                base_norm = a_vec.float().norm(dim=-1).mean().item()
                row[k]["base_norm"] = base_norm
                row[k]["cos_ctx_base"] = kvdw.cosine(b_vec, a_vec).mean().item()
                row[k]["cos_dw_base"] = kvdw.cosine(c_vec, a_vec).mean().item()
                row[k]["v_ctx_minus_v_dw_norm"] = (b_vec.float() - c_vec.float()).norm(dim=-1).mean().item()
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
            continue

    per_site = {k: {m: accum[k][m] / max(1, n_probes) for m in accum[k]} for k in site_keys}
    (out_dir / "kvdw.json").write_text(
        json.dumps({"context": f"ctxonly_{args.context}", "n_probes": n_probes,
                    "n_sites": len(per_site), "per_site": per_site}, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote kvdw.json ({n_probes} probes, {len(per_site)} sites)")

    del student, base
    torch.cuda.empty_cache()

    # ============================================================
    # Phase 4b: ΔW Frob between ctxonly student and base
    # ============================================================
    print(f"\n===== Phase 4b ΔW Frob =====")
    import re
    LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
    GROUP_PATTERNS = {
        "attn.q_proj": re.compile(r"self_attn\.q_proj\.weight$"),
        "attn.k_proj": re.compile(r"self_attn\.k_proj\.weight$"),
        "attn.v_proj": re.compile(r"self_attn\.v_proj\.weight$"),
        "attn.o_proj": re.compile(r"self_attn\.o_proj\.weight$"),
        "mlp.gate_proj": re.compile(r"mlp\.gate_proj\.weight$"),
        "mlp.up_proj": re.compile(r"mlp\.up_proj\.weight$"),
        "mlp.down_proj": re.compile(r"mlp\.down_proj\.weight$"),
        "input_layernorm": re.compile(r"input_layernorm\.weight$"),
        "post_attention_layernorm": re.compile(r"post_attention_layernorm\.weight$"),
    }

    @torch.no_grad()
    def _delta():
        base_sd = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16).state_dict()
        trained_sd = AutoModelForCausalLM.from_pretrained(str(save_dir), dtype=torch.bfloat16).state_dict()
        by_layer = defaultdict(lambda: {"frob_sq": 0.0, "n_params": 0})
        by_group = defaultdict(float)
        by_lg = defaultdict(float)
        total_sq = 0.0
        total = 0
        for name, b in base_sd.items():
            if name not in trained_sd:
                continue
            t = trained_sd[name]
            if b.shape != t.shape:
                continue
            d = (t.float() - b.float())
            sq = (d * d).sum().item()
            n = d.numel()
            total_sq += sq; total += n
            m = LAYER_RE.search(name)
            layer = int(m.group(1)) if m else None
            if layer is not None:
                by_layer[layer]["frob_sq"] += sq
                by_layer[layer]["n_params"] += n
            for grp, pat in GROUP_PATTERNS.items():
                if pat.search(name):
                    by_group[grp] += sq
                    if layer is not None:
                        by_lg[(layer, grp)] += sq
                    break
        per_layer = [{
            "layer": L,
            "frob": s["frob_sq"] ** 0.5,
            "frob_per_param": (s["frob_sq"] / max(1, s["n_params"])) ** 0.5,
            "n_params": s["n_params"],
        } for L, s in sorted(by_layer.items())]
        return {
            "total_frob": total_sq ** 0.5,
            "total_params": total,
            "by_group": {g: v ** 0.5 for g, v in by_group.items()},
            "per_layer": per_layer,
            "per_layer_group": [{"layer": L, "group": g, "frob": v ** 0.5}
                                for (L, g), v in sorted(by_lg.items())],
        }

    res = _delta()
    (out_dir / "delta_norms.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  total Frob: {res['total_frob']:.4f}")

    print("\n===== ALL DONE =====")


if __name__ == "__main__":
    main()
