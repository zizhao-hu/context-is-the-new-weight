"""Render the three-way comparison: in-context vs context-distilled vs prefix-tuned.

Reads outputs/01_synth_distill_kvdw/prefix_<context>/prefix_activations.json
for every context with a trained prefix and produces:
  fig_prefix_cos_per_layer.png    — three pairwise cosines per layer, one panel per context
  fig_prefix_norms_per_layer.png  — three magnitudes per layer, one panel per context
  fig_prefix_summary_table.csv    — per-context peak cosines for each pair
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


CONTEXTS = ["haiku", "pirate", "concise", "fewshot_translate_fr", "factual"]


def load(out_root: Path, ctx: str, split: str = "train"):
    p = out_root / f"prefix_{ctx}" / "prefix_activations.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get(split)


def render_cosines(out_root: Path, fig_path: Path, split: str = "train"):
    n = len(CONTEXTS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, ctx in zip(axes, CONTEXTS):
        data = load(out_root, ctx, split)
        if data is None:
            ax.set_title(f"{ctx} (no data)")
            continue
        pl = data["per_layer"]
        x = [r["layer"] for r in pl]
        ax.plot(x, [r["cos_ctx_dist"] for r in pl], color="tab:red", marker=".", label="cos(v_ctx, v_dist)")
        ax.plot(x, [r["cos_ctx_prefix"] for r in pl], color="tab:green", marker=".", label="cos(v_ctx, v_prefix)")
        ax.plot(x, [r["cos_dist_prefix"] for r in pl], color="tab:blue", marker=".", linestyle="--", alpha=0.7,
                label="cos(v_dist, v_prefix)")
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.5)
        ax.set_title(ctx)
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("cosine")
    plt.suptitle(f"Pairwise cosines: in-context vs context-distilled vs prefix-tuned ({split.upper()} queries)")
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[render] wrote {fig_path}")


def render_norms(out_root: Path, fig_path: Path, split: str = "train"):
    n = len(CONTEXTS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, ctx in zip(axes, CONTEXTS):
        data = load(out_root, ctx, split)
        if data is None:
            ax.set_title(f"{ctx} (no data)")
            continue
        pl = data["per_layer"]
        x = [r["layer"] for r in pl]
        ax.plot(x, [r["v_ctx_norm"] for r in pl], color="tab:red", marker=".", label="||v_ctx||")
        ax.plot(x, [r["v_dist_norm"] for r in pl], color="tab:blue", marker=".", linestyle="--", label="||v_dist||")
        ax.plot(x, [r["v_prefix_norm"] for r in pl], color="tab:green", marker=".", label="||v_prefix||")
        ax.set_title(ctx)
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("L2 norm")
    plt.suptitle(f"Activation magnitudes vs base ({split.upper()} queries)")
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[render] wrote {fig_path}")


def write_summary(out_root: Path, csv_path: Path, split: str = "train"):
    rows = []
    for ctx in CONTEXTS:
        data = load(out_root, ctx, split)
        if data is None:
            continue
        pl = data["per_layer"]
        peak_dist = max(pl, key=lambda r: r["cos_ctx_dist"])
        peak_prefix = max(pl, key=lambda r: r["cos_ctx_prefix"])
        rows.append({
            "context": ctx,
            "peak_cos_ctx_dist": round(peak_dist["cos_ctx_dist"], 3),
            "peak_layer_dist": peak_dist["layer"],
            "peak_cos_ctx_prefix": round(peak_prefix["cos_ctx_prefix"], 3),
            "peak_layer_prefix": peak_prefix["layer"],
            "delta_prefix_minus_dist": round(peak_prefix["cos_ctx_prefix"] - peak_dist["cos_ctx_dist"], 3),
        })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[render] wrote {csv_path}")
    print()
    print(f"{'context':<22} {'cos(ctx,dist)':>13} {'cos(ctx,prefix)':>15} {'delta':>8}")
    for r in rows:
        print(f"{r['context']:<22} {r['peak_cos_ctx_dist']:>13.3f} {r['peak_cos_ctx_prefix']:>15.3f} "
              f"{r['delta_prefix_minus_dist']:>+8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    fig_dir = Path(args.fig_dir)

    for split in ("train", "val", "ood"):
        render_cosines(out_root, fig_dir / f"fig_prefix_cos_{split}.png", split=split)
        render_norms(out_root, fig_dir / f"fig_prefix_norms_{split}.png", split=split)
    write_summary(out_root, fig_dir / "prefix_summary.csv", split="train")


if __name__ == "__main__":
    main()
