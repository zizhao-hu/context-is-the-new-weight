#!/usr/bin/env python3
"""Method figure for Finding 2: the two controls on the loss.

Truncate (top): how many context rows go unscored on one chunk (C=8, W=4). The attention mask is
the same sliding band in all four; only the loss changes. Panel 1 scores every query, including the
starved rows at the chunk start. Panel 4 scores only queries with a full window, so every scored
query sees exactly W real tokens; that endpoint is ST-SWA.

Mix (bottom): instead of one truncation everywhere, alternate panel-1 steps with panel-4 steps at
ratio alpha. The rows show 3:1 and 1:3, i.e. alpha 0.25 and 0.75, each step drawn as its own mask.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.lines import Line2D
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
PANELS = [(0, "1. SWA", "$0$ skipped"), (1, "2. T-SWA", "$1$ skipped"),
          (2, "3. T-SWA", "$2$ skipped"), (W - 1, "4. ST-SWA", "$W{-}1$ skipped")]


def mask(ax, ox, oy, skip, cell=1.0, lw=0.3, frame=0.8):
    """the sliding band on one chunk; rows before `skip` are attended but not scored"""
    for q in range(Q):
        for c in range(Q):
            if q - W < c <= q:
                ax.add_patch(Rectangle((ox + c * cell, oy + (Q - 1 - q) * cell), cell, cell,
                                       facecolor=BLUE if q >= skip else PALE,
                                       edgecolor="black", lw=lw))
    ax.add_patch(Rectangle((ox, oy), Q * cell, Q * cell, fill=False,
                           edgecolor="black", lw=frame, zorder=5))


fig = plt.figure(figsize=(3.20, 1.93))
gs = fig.add_gridspec(3, 4, height_ratios=[2.45, 0.75, 1.45],
                      hspace=0.04, wspace=0.10,
                      left=0.012, right=0.995, top=0.90, bottom=0.02)

for i, (skip, title, sub) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, i])
    mask(ax, 0, 0, skip)
    ax.plot([-1.05, -1.05], [0, Q - skip], color=DK, lw=1.2, clip_on=False)   # scored block
    ax.text(Q / 2.0, -0.55, sub, ha="center", va="top", fontsize=FS - 0.8, color="0.35")
    if i == 0:                                   # name the bracket once
        ax.text(-1.95, Q / 2.0, "loss", rotation=90, ha="center", va="center",
                fontsize=FS - 0.6, color=DK, fontweight="bold")
    ax.set_xlim(-2.6, Q + 0.15); ax.set_ylim(-2.2, Q + 0.15)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=FS, pad=1.5, color="0.2")

arr = fig.add_subplot(gs[1, :]); arr.axis("off")
arr.set_xlim(0, 1); arr.set_ylim(0, 1)
arr.add_patch(FancyArrow(0.0, 0.5, 1.0, 0, width=0.028, head_width=0.22, head_length=0.02,
                         length_includes_head=True, color="0.35"))

GAP, SPLIT = 2.2, 7.0
bx = fig.add_subplot(gs[2, :])                     # one axes, so the row fills the width
step = Q + GAP
groups = [(0.25, r"$\alpha{=}0.25$"), (0.75, r"$\alpha{=}0.75$")]
xs = []
x = 0.0
for gi, (a, lab) in enumerate(groups):
    n4 = int(round(4 * a))                                   # steps taken with panel 4
    order = [0] * (4 - n4) + [W - 1] * n4
    x0 = x
    for skip in order:
        mask(bx, x, 0, skip, cell=1.0, lw=0.12, frame=0.45)
        x += step
    bx.text(x0 + (4 * step - GAP) / 2.0, -2.2, lab, ha="center", va="top",
            fontsize=FS, color="0.2")
    xs.append(x - GAP)
    if gi == 0:
        bx.plot([x - GAP + SPLIT / 2, x - GAP + SPLIT / 2], [-1.2, Q + 0.6],
                color="0.75", lw=0.9, ls=":")
        x += SPLIT
bx.set_xlim(-0.4, x - GAP + 0.4); bx.set_ylim(-4.2, Q + 0.7)
bx.set_aspect("equal"); bx.axis("off")

_pa = arr.get_position()
_ay = _pa.y0 + 0.5 * (_pa.y1 - _pa.y0)          # the arrow line itself, not the axes edge
fig.text(_pa.x0, _ay + 0.012, "truncate", ha="left", va="bottom",
         fontsize=FS, color=DK, fontweight="bold")
fig.text(_pa.x0, _ay - 0.012, "mix", ha="left", va="top",
         fontsize=FS, color=DK, fontweight="bold")
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
