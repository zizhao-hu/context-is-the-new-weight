#!/usr/bin/env python3
"""b (naive SWA) vs OURS (constant-context sliding): persistent-column decomposition,
split by query regime. Same criterion as tab:sinkdecomp / fig:sinkbars (>5x uniform).

Identical mask, W, data and loss-token budget -- the only difference is which queries get
loss, hence how many SCORED queries have pos-0 in their receptive field (naive: W; ours: 1).
COLD (q<W) is where pos-0 is visible; DEEP (q>=W) has pos-0 structurally evicted (p0 == 0).
Freed pos-0 mass mostly RELOCATES onto separators rather than returning to content.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (label, p0, sep, content) -- measured, sinkdecomp_bf.py, W=128 C=256 n=32
ROWS = [
    ("a. triangle",        .2491, .0000, .0498, .7011),
    ("b. naive SWA",       .1197, .0000, .0743, .8061),
    ("e. cc sliding",      .0554, .0000, .1136, .8310),
    ("   $+$ sink token",  .0146, .3501, .0348, .6005),
    ("   $+$ sink prefix", .0244, .4982, .0247, .4526),
    ("   $+$ sink scalar", .0307, .3689, .0400, .5605),
    None,
    ("a. triangle",        .1676, .0000, .1574, .6750),
    ("b. naive SWA",       .0000, .0000, .2289, .7711),
    ("e. cc sliding",      .0000, .0000, .2202, .7798),
    ("   $+$ sink token",  .0000, .3165, .1195, .5641),
    ("   $+$ sink prefix", .0000, .4640, .0873, .4487),
    ("   $+$ sink scalar", .0000, .3322, .1173, .5505),
]
C_P0, C_REG, C_SEP, C_CT = "#4C72B0", "#DD5B45", "#55A868", "#D3D3D3"

labels, vals, ys = [], [], []
y = 0.0
for r in ROWS:
    if r is None:
        y += 0.8; continue
    labels.append(r[0]); vals.append(r[1:]); ys.append(y); y += 1.0
vals = np.array(vals); ys = np.array(ys); ys = ys.max() - ys
p0, reg, sep, ct = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7})
fig, ax = plt.subplots(figsize=(6.8, 5.0))
h = 0.68
ax.barh(ys, p0,  h, color=C_P0,  edgecolor="white", linewidth=0.6)
ax.barh(ys, reg, h, left=p0, color=C_REG, edgecolor="white", linewidth=0.6)
ax.barh(ys, sep, h, left=p0 + reg, color=C_SEP, edgecolor="white", linewidth=0.6)
ax.barh(ys, ct,  h, left=p0 + reg + sep, color=C_CT, edgecolor="white", linewidth=0.6)
def lab(x, yy, v, sz=7.6):
    ax.text(x, yy, ".%02d" % round(v * 100), ha="center", va="center",
            fontsize=sz, color="white", fontweight="bold")
for yy, a_, r_, b_ in zip(ys, p0, reg, sep):
    if a_ >= 0.045: lab(a_ / 2, yy, a_)
    if r_ >= 0.045: lab(a_ + r_ / 2, yy, r_, 8.2)
    if b_ >= 0.045: lab(a_ + r_ + b_ / 2, yy, b_)

ax.text(-0.42, (ys[0] + ys[5]) / 2, "COLD\n$q<W$", ha="center", va="center",
        fontsize=8.5, fontweight="bold")
ax.text(-0.42, (ys[6] + ys[11]) / 2, "DEEP\n$q\\geq W$", ha="center", va="center",
        fontsize=8.5, fontweight="bold", color="#777")
ax.set_xlim(0, 1.0); ax.set_ylim(-0.7, ys.max() + 0.7)
ax.set_xlabel("attention mass", fontsize=9)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xticks(np.arange(0, 1.01, 0.25)); ax.tick_params(length=3, labelsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(handles=[Patch(facecolor=C_P0, label="p0 sink"),
                   Patch(facecolor=C_REG, label="trainable sink"),
                   Patch(facecolor=C_SEP, label="separator sink"),
                   Patch(facecolor=C_CT, label="content")],
          loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4, frameon=False,
          fontsize=8, handlelength=1.0, columnspacing=1.1, handletextpad=0.4)
plt.tight_layout(rect=(0, 0, 1, 0.99))
out = "paper/attention-sink/figures/bf_sinkbars.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
