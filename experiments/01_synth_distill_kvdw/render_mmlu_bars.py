"""Render the MMLU-Pro 3-way accuracy bar chart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONTEXTS = ["haiku", "pirate", "concise", "fewshot_translate_fr", "factual",
            "conv_history", "compressed_history"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--fig-path", default="figures/fig_mmlu_bars.png")
    args = ap.parse_args()
    out_root = Path(args.out_root)

    rows = []
    for ctx in CONTEXTS:
        p = out_root / ctx / "mmlu_pro_eval.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "context": ctx,
            "base": d["base"]["accuracy"],
            "in_context": d["in_context"]["accuracy"],
            "distilled": d["distilled"]["accuracy"],
        })
    if not rows:
        print("no MMLU-Pro data")
        return

    labels = [r["context"] for r in rows]
    base = [r["base"] for r in rows]
    in_ctx = [r["in_context"] for r in rows]
    dist = [r["distilled"] for r in rows]

    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - width, base, width, label="base", color="#888")
    ax.bar(x,         in_ctx, width, label="in-context", color="tab:blue")
    ax.bar(x + width, dist,   width, label="context-distilled", color="tab:red")

    # 0.10 is rough random baseline for 10-option MCQ
    ax.axhline(0.10, color="black", linestyle=":", linewidth=0.7, alpha=0.5, label="random (10-option)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("MMLU-Pro accuracy")
    ax.set_title("MMLU-Pro accuracy by model variant — context-distilled destroys general capability")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    Path(args.fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.fig_path, dpi=150)
    plt.close()
    print(f"wrote {args.fig_path}")


if __name__ == "__main__":
    main()
