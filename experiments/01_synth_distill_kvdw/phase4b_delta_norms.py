"""Phase 4b — per-layer ΔW Frobenius norm analysis.

For each trained student `θ_C`, compute ‖θ_C − θ_0‖_F per (layer, parameter
group) and dump to JSON. This is *complementary* to phase4_kvdw.py's
alignment curve: alignment tells us *where the change has the same effect*
as context; Frobenius norm tells us *where the weights actually moved*.

Usage (from repo root):
  python experiments/01_synth_distill_kvdw/phase4b_delta_norms.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --trained-dir /scratch1/.../saves/haiku \
    --out outputs/01_synth_distill_kvdw/haiku/delta_norms.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


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


def _classify(param_name: str) -> tuple[int | None, str | None]:
    m = LAYER_RE.search(param_name)
    layer = int(m.group(1)) if m else None
    for grp, pat in GROUP_PATTERNS.items():
        if pat.search(param_name):
            return layer, grp
    return layer, None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--trained-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[delta_norms] loading base from {args.base_model}")
    base_sd = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).state_dict()
    print(f"[delta_norms] loading trained from {args.trained_dir}")
    trained_sd = AutoModelForCausalLM.from_pretrained(args.trained_dir, dtype=torch.bfloat16).state_dict()

    per_param: dict[str, dict] = {}
    by_layer: dict = defaultdict(lambda: {"frob_sq": 0.0, "n_params": 0})
    by_group: dict = defaultdict(float)
    by_layer_group: dict = defaultdict(float)
    total_frob_sq = 0.0
    total_params = 0

    for name, base_w in base_sd.items():
        if name not in trained_sd:
            continue
        trained_w = trained_sd[name]
        if base_w.shape != trained_w.shape:
            continue
        delta = (trained_w.float() - base_w.float())
        frob_sq = (delta * delta).sum().item()
        n = delta.numel()
        per_param[name] = {
            "frob": frob_sq ** 0.5,
            "max_abs": delta.abs().max().item(),
            "shape": tuple(delta.shape),
        }

        layer, grp = _classify(name)
        total_frob_sq += frob_sq
        total_params += n
        if layer is not None:
            by_layer[layer]["frob_sq"] += frob_sq
            by_layer[layer]["n_params"] += n
            if grp is not None:
                by_layer_group[(layer, grp)] += frob_sq
                by_group[grp] += frob_sq

    per_layer = []
    for layer in sorted(by_layer):
        s = by_layer[layer]
        per_layer.append({
            "layer": layer,
            "frob": s["frob_sq"] ** 0.5,
            "frob_per_param": (s["frob_sq"] / max(1, s["n_params"])) ** 0.5,
            "n_params": s["n_params"],
        })

    per_layer_group = []
    for (layer, grp), v in sorted(by_layer_group.items()):
        per_layer_group.append({"layer": layer, "group": grp, "frob": v ** 0.5})

    result = {
        "total_frob": total_frob_sq ** 0.5,
        "total_params": total_params,
        "by_group": {g: v ** 0.5 for g, v in by_group.items()},
        "per_layer": per_layer,
        "per_layer_group": per_layer_group,
        # per-param dict is large; store only top-32 by frob to keep file small
        "top32_params": sorted(
            ({"name": k, **v} for k, v in per_param.items()),
            key=lambda r: -r["frob"],
        )[:32],
    }

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[delta_norms] wrote {out_path}")
    print(f"[delta_norms] total ΔW Frobenius = {result['total_frob']:.4f} over {total_params:,} parameters")


if __name__ == "__main__":
    main()
