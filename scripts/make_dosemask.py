#!/usr/bin/env python3
"""Method figure for Finding 2: the dose is how many context rows go unscored.

Four panels of the same sliding band on one chunk (C=6, W=3), differing only in which query rows
carry loss. b scores every row, including the starved ones at the chunk start. e scores only rows
with a full window. Between them sit partial doses, so the sweep in the ablation figures is one
axis, not two regimes.

Colour follows regimes.png: filled cells are attended, blue rows are scored, pale rows are
context-only. The bracket marks the scored block.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
figstyle.apply()

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/dose_mask.png")

BLUE = (0.18, 0.43, 0.71)          # scored row, in window
PALE = (0.80, 0.86, 0.93)          # context-only row, in window
DK = "#333"
Q = 6                              # chunk length C
W = 3                              # window
PANELS = [(0, "b. SWA\n$0$ unscored"),
          (1, "$1$ unscored"),
          (2, "$2$ unscored"),
          (W - 1, "e. T-SWA\n$W{-}1$ unscored")]

fig, axes = plt.subplots(1, len(PANELS), figsize=(figstyle.COL, 1.30))
for ax, (skip, title) in zip(axes, PANELS):
    for q in range(Q):
        y = Q - 1 - q
        for c in range(Q):
            if not (q - W < c <= q):
                continue                                   # outside the sliding band
            ax.add_patch(Rectangle((c, y), 1, 1, facecolor=BLUE if q >= skip else PALE,
                                   edgecolor="black", lw=0.35))
    ax.add_patch(Rectangle((0, 0), Q, Q, fill=False, edgecolor="black", lw=0.9, zorder=5))
    ytop = Q - skip
    ax.plot([-0.55, -0.55], [0, ytop], color=DK, lw=1.3, clip_on=False)   # scored block
    ax.set_xlim(-1.0, Q + 0.15); ax.set_ylim(-0.15, Q + 0.15)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=figstyle.FS_TICK - 0.6, pad=1.5, color="0.2", linespacing=1.15)

axes[0].text(-1.6, Q / 2.0, "loss", rotation=90, ha="center", va="center",
             fontsize=figstyle.FS_TICK - 0.6, color=DK, fontweight="bold")
plt.tight_layout(w_pad=0.15)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
