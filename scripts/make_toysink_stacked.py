#!/usr/bin/env python3
"""Finding-1 support figure (toy_sink.png), built exactly like make_a_sinkbars.py.

Same construction as that figure so the two read as a pair: one axes carrying two blocks
separated by a dotted divider, the block names under the axis rather than as subplot titles,
a boxed legend overlaid along the top band, and the shared figstyle sizes.

Left block: from-scratch BPE toy GPT (sink-head attention, W=128), full causal vs SWA.
Right block: continued pretraining of Llama-3.2-3B (mean head, W=512, n=12), base / full / SWA.
Stacked classes per model: p0 sink + distributed sink + content (remainder to 1), +-1 SEM caps
at the segment edges. Numbers from scripts/make_tri_2panel.py (measured; toy sink-head, CPT
mean-head).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os as _o
sys.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import figstyle
figstyle.apply()
from matplotlib.patches import Patch

OUT = _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                   "paper/attention-sink/figures/toy_sink.png")

# (label, p0, p0sem, dist, distsem) -- W=128 BPE toy, FULL-CAUSAL eval (both models given
# access to p0), all-heads means, sequence within max context (fulldecomp job 5166394).
TOY = [("full", 0.2491, 0.0032, 0.0498, 0.0017),
       ("SWA",  0.1197, 0.0019, 0.0713, 0.0022)]
CPT = [("base", 0.5855, 0.0031, 0.0479, 0.0011),
       ("full", 0.6404, 0.0046, 0.0537, 0.0018),
       ("SWA",  0.1630, 0.0267, 0.5017, 0.0311)]

C_P0, C_SEP, C_CT = "#4C72B0", "#55A868", "#D3D3D3"

# one axes with two blocks and a dotted divider, exactly like the a_sinkbars figure
GAP = 1.35
xL = np.arange(len(TOY), dtype=float)
xR = xL[-1] + 1 + GAP + np.arange(len(CPT), dtype=float)
fig, ax = plt.subplots(figsize=(figstyle.COL, 1.52))
w = 0.80

for xs, rows in ((xL, TOY), (xR, CPT)):
    p0 = np.array([r[1] for r in rows]); p0s = np.array([r[2] for r in rows])
    cm = np.array([r[3] for r in rows]); cms = np.array([r[4] for r in rows])
    ct = 1.0 - p0 - cm
    ax.bar(xs, p0, w, color=C_P0, edgecolor="black", linewidth=0.6, zorder=3)
    ax.bar(xs, cm, w, bottom=p0, color=C_SEP, edgecolor="black", linewidth=0.6, zorder=3)
    ax.bar(xs, ct, w, bottom=p0 + cm, color=C_CT, edgecolor="black", linewidth=0.6, zorder=3)
    edges = np.stack([p0, p0 + cm], axis=1)
    sv = np.stack([p0s, cms], axis=1)
    for j in range(2):
        ax.errorbar(xs, edges[:, j], yerr=sv[:, j], fmt="none", ecolor="0.15",
                    elinewidth=0.7, capsize=1.2, capthick=0.7, zorder=6)
    # no value chips: the thin segments here (0.05 to 0.07) cannot hold one without colliding
    # with the chip on the segment below, and every cited number is already in the prose

div = (xL[-1] + xR[0]) / 2.0
ax.axvline(div, color="0.75", lw=0.9, ls=":", zorder=1)
ax.set_ylim(0, 1.30)
ax.set_yticks(np.arange(0, 1.01, 0.25))
ax.set_xticks(np.concatenate([xL, xR]))
ax.set_xticklabels([r[0] for r in TOY] + [r[0] for r in CPT],
                   fontsize=figstyle.FS_TICK - 0.8)
ax.set_xlim(xL[0] - 0.75, xR[-1] + 0.75)
figstyle.clean(ax)
ax.text(np.mean(xL), -0.22, "from scratch", ha="center", va="top",
        fontsize=figstyle.FS_AXIS, transform=ax.get_xaxis_transform())
ax.text(np.mean(xR), -0.22, "continued pretraining", ha="center", va="top",
        fontsize=figstyle.FS_AXIS, transform=ax.get_xaxis_transform())

ax.legend(handles=[Patch(facecolor=C_P0, edgecolor="black", lw=0.6, label="p$0$ sink"),
                   Patch(facecolor=C_SEP, edgecolor="black", lw=0.6, label="distributed sink"),
                   Patch(facecolor=C_CT, edgecolor="black", lw=0.6, label="content")],
          loc="upper center", ncol=3, frameon=True, fancybox=False,
          fontsize=figstyle.FS_LEGEND, handlelength=0.9, handleheight=0.8, columnspacing=0.8,
          handletextpad=0.3, labelspacing=0.2, borderpad=0.25, borderaxespad=0.15,
          edgecolor="0.4", framealpha=1.0)
figstyle.yname(ax, "attention mass", pad=0.055)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
