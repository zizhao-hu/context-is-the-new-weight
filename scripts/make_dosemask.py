#!/usr/bin/env python3
"""Method figure for Finding 2: the two controls on the loss.

Top row, the dose: how many context rows go unscored on one chunk (C=8, W=4). The attention mask
is the same sliding band in all four; only the loss changes. Panel 1 scores every query, including
the starved rows at the chunk start. Panel 4 scores only queries with a full window, so every
scored query sees exactly W real tokens; that endpoint is ST-SWA.

Bottom row, the mix: instead of one dose everywhere, alternate panel-1 steps with panel-4 steps at
ratio alpha. The stacks show 3:1 and 1:3, i.e. alpha 0.25 and 0.75.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
figstyle.apply()

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/dose_mask.png")

BLUE = (0.18, 0.43, 0.71)          # scored row
PALE = (0.80, 0.86, 0.93)          # attended but not scored
DK = "#333"
Q, W = 8, 4
FS = figstyle.FS_TICK - 0.6
PANELS = [(0, "1. SWA"), (1, "2. T-SWA"), (2, "3. T-SWA"), (W - 1, "4. ST-SWA")]

fig = plt.figure(figsize=(figstyle.COL, 1.95))
gs = fig.add_gridspec(3, 4, height_ratios=[2.5, 0.22, 1.30], hspace=0.02, wspace=0.16)

for i, (skip, title) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, i])
    for q in range(Q):
        for c in range(Q):
            if q - W < c <= q:
                ax.add_patch(Rectangle((c, Q - 1 - q), 1, 1,
                                       facecolor=BLUE if q >= skip else PALE,
                                       edgecolor="black", lw=0.3))
    ax.add_patch(Rectangle((0, 0), Q, Q, fill=False, edgecolor="black", lw=0.8, zorder=5))
    ax.plot([-0.5, -0.5], [0, Q - skip], color=DK, lw=1.2, clip_on=False)     # scored block
    ax.set_xlim(-1.0, Q + 0.1); ax.set_ylim(-0.1, Q + 0.1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=FS, pad=1.5, color="0.2")

# the dose axis
arr = fig.add_subplot(gs[1, :]); arr.axis("off")
arr.set_xlim(0, 1); arr.set_ylim(0, 1)
arr.add_patch(FancyArrow(0.02, 0.5, 0.94, 0, width=0.06, head_width=0.5, head_length=0.022,
                         length_includes_head=True, color="0.35"))

# the mix: alternate panel 1 and panel 4 across steps
for k, (a, lab) in enumerate(((0.25, r"$\alpha{=}0.25$"), (0.75, r"$\alpha{=}0.75$"))):
    bx = fig.add_subplot(gs[2, 2 * k:2 * k + 2])
    n4 = int(round(4 * a))                                   # steps taken with panel 4
    order = [1] * (4 - n4) + [4] * n4
    for j, which in enumerate(order):
        bx.add_patch(Rectangle((j, 0), 0.86, 1, facecolor=BLUE if which == 4 else PALE,
                               edgecolor="black", lw=0.4))
        bx.text(j + 0.43, 0.5, str(which), ha="center", va="center", fontsize=FS - 0.4,
                color="white" if which == 4 else "0.25", fontweight="bold")
    bx.text(2.0, -0.30, lab, ha="center", va="top", fontsize=FS, color="0.2")
    bx.set_xlim(-0.25, 4.1); bx.set_ylim(-1.0, 1.1)
    bx.set_aspect("equal"); bx.axis("off")

fig.text(0.008, 0.20, "mix", ha="left", va="center", fontsize=FS, color=DK, fontweight="bold")
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
