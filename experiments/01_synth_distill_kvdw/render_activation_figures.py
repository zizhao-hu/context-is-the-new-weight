"""Render the 4 per-layer activation-delta figures.

Reads outputs/01_synth_distill_kvdw/<context>/activations_{train,val}.json
for all 5 contexts and produces:
  fig1_delta_vs_base_train.png   — ||v_ctx|| and ||v_dw|| per layer, train queries
  fig2_delta_vs_base_val.png     — same on validation queries
  fig3_delta_ctx_vs_dw_train.png — ||v_ctx - v_dw|| per layer, train queries
  fig4_delta_ctx_vs_dw_val.png   — same on validation queries

Usage:
  python experiments/01_synth_distill_kvdw/render_activation_figures.py \
    --out-root outputs/01_synth_distill_kvdw \
    --fig-dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


CONTEXTS = ["haiku", "pirate", "concise", "fewshot_translate_fr", "factual"]


def load_per_layer(out_root: Path, ctx: str, split: str) -> list[dict] | None:
    p = out_root / ctx / f"activations_{split}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["per_layer"]


def render_delta_vs_base(out_root: Path, fig_path: Path, split: str):
    plt.figure(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    for i, ctx in enumerate(CONTEXTS):
        rows = load_per_layer(out_root, ctx, split)
        if rows is None:
            print(f"[render] missing {split} data for {ctx}")
            continue
        x = [r["layer"] for r in rows]
        y_ctx = [r["v_ctx_norm"] for r in rows]
        y_dw = [r["v_dw_norm"] for r in rows]
        c = cmap(i)
        plt.plot(x, y_ctx, color=c, linestyle="-", marker=".", label=f"{ctx} v_ctx (S1)")
        plt.plot(x, y_dw, color=c, linestyle="--", marker=".", label=f"{ctx} v_dw (S2)")
    plt.xlabel("layer")
    plt.ylabel(r"$\|\Delta \text{activation}\|$ vs base (mean over attn+mlp at each layer)")
    plt.title(f"Per-layer activation delta vs base — {split.upper()} queries\n"
              f"solid = in-context (v_ctx), dashed = context-distillation-FT (v_dw)")
    plt.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[render] wrote {fig_path}")


def render_ctx_vs_dw(out_root: Path, fig_path: Path, split: str):
    plt.figure(figsize=(10, 4.5))
    for ctx in CONTEXTS:
        rows = load_per_layer(out_root, ctx, split)
        if rows is None:
            continue
        x = [r["layer"] for r in rows]
        y = [r["v_ctx_minus_v_dw_norm"] for r in rows]
        plt.plot(x, y, marker=".", label=ctx)
    plt.xlabel("layer")
    plt.ylabel(r"$\|v_\mathrm{ctx} - v_\mathrm{dw}\|$  (mean over attn+mlp)")
    plt.title(f"Per-layer disagreement between in-context and context-distillation-FT — {split.upper()} queries")
    plt.legend(fontsize=9)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[render] wrote {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    fig_dir = Path(args.fig_dir)

    render_delta_vs_base(out_root, fig_dir / "fig1_delta_vs_base_train.png", "train")
    render_delta_vs_base(out_root, fig_dir / "fig2_delta_vs_base_val.png", "val")
    render_ctx_vs_dw(out_root, fig_dir / "fig3_delta_ctx_vs_dw_train.png", "train")
    render_ctx_vs_dw(out_root, fig_dir / "fig4_delta_ctx_vs_dw_val.png", "val")


if __name__ == "__main__":
    main()
