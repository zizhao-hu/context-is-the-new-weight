#!/usr/bin/env python3
"""SWA -> S-SWA loss-mask ablation as BAR charts with trend curves.

Left: sink distribution per dose -- stacked bars (p0 + separator) with trend curves
      tracing the p0 drain and the total.
Right: perplexity per dose -- bars for short-context ppl (the rising cost) with its trend
       curve, and a flat line+band for deep streaming ppl (the axis that never moves).
Data: figures/sf_ablation.tsv (measured; refuses to plot without it).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DATA = "figures/sf_ablation.tsv"
OUT = "paper/attention-sink/figures/sf_ablation.png"
if not os.path.exists(DATA):
    sys.exit("missing %s" % DATA)

rows = [[float(x) for x in l.split("\t")] for l in open(DATA) if l.strip() and not l.startswith("#")]
rows.sort()
sf, ppl, ppl_sem, p0, p0s, sep, seps, short = map(np.array, zip(*rows))
tot = p0 + sep
x = np.arange(len(sf))
labels = ["%d" % v for v in sf]

C_P0, C_SEP, C_SHORT, C_STREAM = "#4C72B0", "#55A868", "#B04C4C", "#555555"
plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.2))

# ---- left: sink distribution ----
w = 0.72
a1.bar(x, p0, w, color=C_P0, edgecolor="white", linewidth=0.6)
a1.bar(x, sep, w, bottom=p0, color=C_SEP, edgecolor="white", linewidth=0.6)
a1.plot(x, p0, "-o", color="#1f3d63", ms=4, lw=1.8, zorder=5)
a1.plot(x, tot, "-o", color="#2e6b45", ms=4, lw=1.8, zorder=5)
a1.errorbar(x, p0, yerr=p0s, fmt="none", ecolor="0.15", elinewidth=0.9, capsize=2, zorder=6)
a1.errorbar(x, tot, yerr=np.sqrt(p0s**2 + seps**2), fmt="none", ecolor="0.15",
            elinewidth=0.9, capsize=2, zorder=6)
a1.set_ylabel("attention mass", fontsize=12)
a1.set_ylim(0, max(tot) * 1.28)
a1.set_title("sink distribution", fontsize=12, fontweight="bold", pad=4)
a1.legend(handles=[Patch(facecolor=C_P0, label="p0 sink"),
                   Patch(facecolor=C_SEP, label="separator sink"),
                   Line2D([], [], color="#1f3d63", marker="o", ms=4, lw=1.8, label="p0 trend"),
                   Line2D([], [], color="#2e6b45", marker="o", ms=4, lw=1.8, label="total trend")],
          frameon=False, fontsize=8.6, ncol=2, loc="upper center",
          handlelength=1.2, columnspacing=0.9, handletextpad=0.4)

# ---- right: perplexity ----
a2.bar(x, short, w, color=C_SHORT, edgecolor="white", linewidth=0.6, alpha=0.85)
a2.plot(x, short, "-o", color="#7a2f2f", ms=4, lw=1.8, zorder=5)
a2.plot(x, ppl, "-o", color=C_STREAM, ms=4, lw=1.8, zorder=5)
a2.fill_between(x, ppl - ppl_sem, ppl + ppl_sem, color=C_STREAM, alpha=0.18, lw=0)
a2.set_ylabel("perplexity", fontsize=12)
a2.set_ylim(0, max(short) * 1.22)
a2.set_title("LM quality", fontsize=12, fontweight="bold", pad=4)
a2.legend(handles=[Patch(facecolor=C_SHORT, label="short (len 64)"),
                   Line2D([], [], color=C_STREAM, marker="o", ms=4, lw=1.8, label="stream (30k)")],
          frameon=False, fontsize=8.6, loc="upper left",
          handlelength=1.2, handletextpad=0.4)

for ax in (a1, a2):
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlabel("unscored context rows", fontsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.5, labelsize=10)
    ax.text(0, -0.24, "SWA", ha="center", fontsize=10, fontweight="bold",
            color="#333", transform=ax.get_xaxis_transform())
    ax.text(len(sf) - 1, -0.24, "S-SWA", ha="center", fontsize=10, fontweight="bold",
            color="#333", transform=ax.get_xaxis_transform())

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
