#!/usr/bin/env python3
"""Finding-1 support figure (toy_sink.png) in the a_sinkbars STACKED style.

Left: from-scratch BPE toy GPT (sink-head attention, W=64) -- full causal vs SWA.
Right: continued pretraining Llama-3.2-3B (mean head, W=512, n=12) -- base / full / SWA.
Stacked classes per model: p0 sink + common-token sink + content (remainder to 1),
white value labels, +-1 SEM caps at segment edges, top legend (make_a_sinkbars.py style).
Numbers from scripts/make_tri_2panel.py (measured; toy sink-head, CPT mean-head).
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

import os as _os
OUT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                   "paper/attention-sink/figures/toy_sink.png")

# (p0, p0sem, dist, distsem) -- W=128 BPE toy, FULL-CAUSAL eval (both models given
# access to p0), all-heads means, sequence within max context (fulldecomp job 5166394).
TOY = [("full", 0.2491, 0.0032, 0.0498, 0.0017),
       ("SWA",  0.1197, 0.0019, 0.0713, 0.0022)]
CPT = [("base", 0.5855, 0.0031, 0.0479, 0.0011),
       ("full", 0.6404, 0.0046, 0.0537, 0.0018),
       ("SWA",  0.1630, 0.0267, 0.5017, 0.0311)]

C_P0, C_SEP, C_CT = "#4C72B0", "#55A868", "#D3D3D3"
fig, axes = plt.subplots(1, 2, figsize=(3.20, 2.35), sharey=True,
                         gridspec_kw={"width_ratios": [2, 3]})

T_L = "from scratch"
T_R = "continued pretraining"
for ax, rows, ttl in ((axes[0], TOY, T_L), (axes[1], CPT, T_R)):
    x = np.arange(len(rows))
    labs = [r[0] for r in rows]
    p0 = np.array([r[1] for r in rows]); p0s = np.array([r[2] for r in rows])
    cm = np.array([r[3] for r in rows]); cms = np.array([r[4] for r in rows])
    ct = 1.0 - p0 - cm
    w = 0.80
    ax.bar(x, p0, w, color=C_P0, edgecolor="white", linewidth=0.6)
    ax.bar(x, cm, w, bottom=p0, color=C_SEP, edgecolor="white", linewidth=0.6)
    ax.bar(x, ct, w, bottom=p0 + cm, color=C_CT, edgecolor="white", linewidth=0.6)
    edges = np.stack([p0, p0 + cm], axis=1)
    sv = np.stack([p0s, cms], axis=1)
    for j in range(2):
        ax.errorbar(x, edges[:, j], yerr=sv[:, j], fmt="none", ecolor="0.15",
                    elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=5)
    for xx, a_, b_ in zip(x, p0, cm):
        for yy, v in ((a_ / 2, a_), (a_ + b_ / 2, b_)):
            if v >= 0.01:
                ax.text(xx, yy, "%.2f" % v, ha="center", va="center", fontsize=figstyle.FS_CHIP,
                        color="0.1", fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.45", lw=0.8, alpha=0.9))
    ax.set_ylim(0, 1.42); ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=figstyle.FS_TICK)
    ax.set_xlim(-0.60, len(rows) - 0.40)
    ax.set_title(ttl, fontsize=figstyle.FS_TICK, fontweight="bold", pad=3)
    ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.tick_params(length=3.0, labelsize=figstyle.FS_TICK)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
figstyle.yname(axes[0], "attention mass", also=(axes[1],), pad=0.105)

plt.tight_layout()
_p0, _p1 = axes[0].get_position(), axes[1].get_position()   # legend inside, along the top band
fig.legend(handles=[Patch(facecolor=C_P0, label="p0 sink"),
                    Patch(facecolor=C_SEP, label="distributed sink"),
                    Patch(facecolor=C_CT, label="content")],
           loc="upper center", bbox_to_anchor=((_p0.x0 + _p1.x1) / 2, _p0.y1 - 0.03),
           bbox_transform=fig.transFigure, ncol=3, frameon=False, fontsize=figstyle.FS_LEGEND,
           handlelength=0.9, handleheight=0.8, columnspacing=0.8, handletextpad=0.4,
           borderpad=0.0, borderaxespad=0.0)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
