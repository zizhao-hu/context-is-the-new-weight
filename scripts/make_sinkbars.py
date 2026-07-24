#!/usr/bin/env python3
"""Stacked-bar view of the persistent-column decomposition (companion to tab:sinkdecomp).
One bar per train regime / trainable-sink design; segments = disjoint deep-query attention
classes (pos-0 sink / trainable sink / distributed separators / ordinary content), summing to 1.
Numbers are the all-heads means from tab:sinkdecomp (BPE toy, W=64, deep queries >W).
Shows the full design matrix: token vs scalar vs prefix, and pos-0 vs riding geometry, on
each base -- the triangle parks at pos-0, the window drains it into separators, and any
trainable sink absorbs whichever sink the mask would otherwise grow (prefix absorbs most)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

GAP = None
# (label, [pos0, trainable_sink, separators]); content = remainder to 1
ROWS = [
    ("a. full-causal",           [.10, .00, .03]),
    ("   $+$ sink token",        [.01, .17, .02]),
    ("   $+$ sink scalar",       [.01, .22, .02]),
    ("   $+$ sink prefix",       [.01, .31, .01]),
    GAP,
    ("b. window (Gemma/SWAT)",   [.00, .00, .09]),
    ("   $+$ sink token",        [.00, .34, .03]),
    ("   $+$ riding sink token", [.00, .39, .03]),
    ("   $+$ sink scalar",       [.00, .41, .03]),
    ("   $+$ sink prefix",       [.00, .51, .02]),
    ("   $+$ riding sink prefix",[.00, .53, .02]),
    GAP,
    ("f. sliding-history",       [.00, .00, .13]),
    ("   $+$ sink token",        [.00, .17, .07]),
    ("   $+$ riding sink token", [.00, .32, .05]),
    GAP,
    ("c. SWAA (sink$+$window)",  [.15, .00, .06]),
    ("g. sliding$+$warmup",      [.00, .00, .12]),
]

C_P0, C_SINK, C_DIST, C_CONT = "#4C72B0", "#DD5B45", "#55A868", "#D3D3D3"

labels, vals, ys = [], [], []
y = 0.0
for r in ROWS:
    if r is GAP:
        y += 0.6
        continue
    labels.append(r[0]); vals.append(r[1]); ys.append(y)
    y += 1.0
vals = np.array(vals)
ys = np.array(ys)
ys = ys.max() - ys                      # flip so first row sits on top
p0, sk, ds = vals[:, 0], vals[:, 1], vals[:, 2]
ct = 1.0 - p0 - sk - ds

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7})
fig, ax = plt.subplots(figsize=(3.4, 4.05))
h = 0.74
ax.barh(ys, p0, h, color=C_P0,   edgecolor="white", linewidth=0.5)
ax.barh(ys, sk, h, left=p0,      color=C_SINK, edgecolor="white", linewidth=0.5)
ax.barh(ys, ds, h, left=p0 + sk, color=C_DIST, edgecolor="white", linewidth=0.5)
ax.barh(ys, ct, h, left=p0 + sk + ds, color=C_CONT, edgecolor="white", linewidth=0.5)

def lab(x, yy, v, size, col="white"):
    ax.text(x, yy, f".{int(round(v*100)):02d}", ha="center", va="center",
            fontsize=size, color=col, fontweight="bold")
for yy, a, b, c in zip(ys, p0, sk, ds):
    if a >= 0.045: lab(a/2, yy, a, 6.6)
    if b >= 0.045: lab(a + b/2, yy, b, 7.0)
    # label the separator sink only where it carries the story (no pos-0 sink crowding it)
    if c >= 0.055 and a < 0.05: lab(a + b + c/2, yy, c, 6.3)

ax.set_xlim(0, 1.0); ax.set_ylim(-0.7, ys.max() + 0.7)
ax.set_xlabel("deep-query attention mass", fontsize=8.5)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=7.4)
ax.set_xticks(np.arange(0, 1.01, 0.25)); ax.tick_params(length=3, labelsize=7.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

handles = [Patch(facecolor=C_P0, label="p0 sink"),
           Patch(facecolor=C_SINK, label="trainable sink"),
           Patch(facecolor=C_DIST, label="distributed (sep.)"),
           Patch(facecolor=C_CONT, label="content")]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
          ncol=2, frameon=False, fontsize=7.2, handlelength=1.0,
          columnspacing=1.0, handletextpad=0.4, labelspacing=0.3)
plt.tight_layout(rect=(0, 0, 1, 0.99))
out = "paper/attention-sink/figures/sink_barchart.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
