#!/usr/bin/env python3
"""Finding 5 figure: passkey retrieval accuracy against depth, full vs sliding deploy.

Measured points from the CARC passkey evals (passkey.log full deploy, native.log sliding
deploy; Llama-3.2-3B CPT pair, L=4096, W=1024, n=20 keys per depth):
  full deploy      a and b both 1.000 at every depth
  sliding W=1024   b: 0,0,0,0,1.0 across depths .1-.9 (retrieves exactly within the last W)
                   a: 0 at every depth (the position-0 anchor is evicted)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
figstyle.apply()

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/passkey.png")

D = [0.1, 0.3, 0.5, 0.7, 0.9]
A_FULL = [1.0] * 5
B_FULL = [1.0] * 5
B_SLIDE = [0.0, 0.0, 0.0, 0.0, 1.0]
A_SLIDE = [0.0] * 5

fig, ax = plt.subplots(figsize=(3.03, 0.92))
ax.axvspan(0.75, 0.97, color="0.92", zorder=0)                       # within the last W
ax.text(0.855, 0.5, "within\nlast $W$", ha="center", va="center",
        fontsize=figstyle.FS_TICK - 0.5, color="0.45")
ax.plot(D, A_FULL, "-o", color=figstyle.C_P0, ms=3.2, lw=1.2, zorder=4)
ax.plot(D, B_FULL, "--s", color=figstyle.C_SEP, ms=3.0, lw=1.2, zorder=4)
ax.plot(D, B_SLIDE, "-^", color=figstyle.C_SEP, ms=3.4, lw=1.2, zorder=5)
ax.plot(D, A_SLIDE, "-v", color=figstyle.C_P0, ms=3.4, lw=1.2, zorder=5)
ax.set_xticks(D)
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_ylim(-0.08, 1.24)
ax.set_xlabel("passkey depth in the $4096$-token context", fontsize=figstyle.FS_AXIS)
figstyle.clean(ax)
figstyle.yname(ax, "retrieval acc.", pad=0.09)
ax.legend(handles=[
    Line2D([], [], color=figstyle.C_P0, marker="o", ms=3.2, lw=1.2, label="$a$ full"),
    Line2D([], [], color=figstyle.C_SEP, marker="s", ls="--", ms=3.0, lw=1.2, label="$b$ full"),
    Line2D([], [], color=figstyle.C_SEP, marker="^", ms=3.4, lw=1.2, label="$b$ sliding"),
    Line2D([], [], color=figstyle.C_P0, marker="v", ms=3.4, lw=1.2, label="$a$ sliding")],
    loc="center left", bbox_to_anchor=(0.13, 0.52), ncol=2, frameon=True, fancybox=False,
    edgecolor="0.4", framealpha=1.0, fontsize=figstyle.FS_LEGEND - 0.5,
    handlelength=1.2, columnspacing=0.7, handletextpad=0.35, borderpad=0.25, labelspacing=0.25)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
