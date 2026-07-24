#!/usr/bin/env python3
"""Finding-2 figure: trainable sinks offload both structural sinks, on BOTH bases.

Left: full-causal base + sink designs (adecomp_5151160.log).
Right: SWA base + sink designs (fulldecomp_5166496.log).
BPE toy, W=128, C=256, within max context, all-heads means, n=32, +-1 SEM.
Stacked classes: p0 sink + trainable sink + distributed sink + content (sum to 1).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/figures/sink_absorb.png"

# (label, p0, p0sem, reg, regsem, sep, sepsem)
FULLFAM = [("full",         0.2491, 0.0032, 0.0000, 0.0000, 0.0498, 0.0017),
           ("+sink token",  0.0255, 0.0004, 0.3283, 0.0040, 0.0387, 0.0016),
           ("+sink scalar", 0.0268, 0.0005, 0.3679, 0.0042, 0.0359, 0.0013),
           ("+sink prefix", 0.0152, 0.0003, 0.4322, 0.0038, 0.0294, 0.0013)]
SWAFAM  = [("SWA",          0.1197, 0.0019, 0.0000, 0.0000, 0.0743, 0.0022),
           ("+sink token",  0.0185, 0.0004, 0.3480, 0.0039, 0.0348, 0.0014),
           ("+sink scalar", 0.0165, 0.0003, 0.4017, 0.0040, 0.0292, 0.0013),
           ("+sink prefix", 0.0090, 0.0002, 0.5238, 0.0042, 0.0203, 0.0011)]

C_P0, C_REG, C_SEP, C_CT = "#4C72B0", "#DD5B45", "#55A868", "#D3D3D3"
plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.65), sharey=True)

for ax, rows, ttl in ((axes[0], FULLFAM, "full-causal base"),
                      (axes[1], SWAFAM, "SWA base")):
    x = np.arange(len(rows))
    labs = [r[0] for r in rows]
    p0 = np.array([r[1] for r in rows]); p0s = np.array([r[2] for r in rows])
    rg = np.array([r[3] for r in rows]); rgs = np.array([r[4] for r in rows])
    sp = np.array([r[5] for r in rows]); sps = np.array([r[6] for r in rows])
    ct = 1.0 - p0 - rg - sp
    w = 0.80
    ax.bar(x, p0, w, color=C_P0, edgecolor="white", linewidth=0.6)
    ax.bar(x, rg, w, bottom=p0, color=C_REG, edgecolor="white", linewidth=0.6)
    ax.bar(x, sp, w, bottom=p0 + rg, color=C_SEP, edgecolor="white", linewidth=0.6)
    ax.bar(x, ct, w, bottom=p0 + rg + sp, color=C_CT, edgecolor="white", linewidth=0.6)
    edges = np.stack([p0, p0 + rg, p0 + rg + sp], axis=1)
    sv = np.stack([p0s, rgs, sps], axis=1)
    for j in range(3):
        ax.errorbar(x, edges[:, j], yerr=sv[:, j], fmt="none", ecolor="0.15",
                    elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=5)
    for xx, a_, b_, c_ in zip(x, p0, rg, sp):
        for yy, v in ((a_ / 2, a_), (a_ + b_ / 2, b_), (a_ + b_ + c_ / 2, c_)):
            if v >= 0.01:
                ax.text(xx, yy, "%.2f" % v, ha="center", va="center", fontsize=9.8,
                        color="0.1", fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.45", lw=0.8, alpha=0.9))
    ax.set_ylim(0, 1); ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=10.8)
    ax.set_xlim(-0.60, len(rows) - 0.40)
    ax.set_title(ttl, fontsize=12, fontweight="bold", pad=4)
    ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.tick_params(length=3.5, labelsize=10.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("attention mass", fontsize=11.6)

plt.tight_layout(rect=(0, 0, 1, 0.90))
_cx = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
fig.legend(handles=[Patch(facecolor=C_P0, label="p0 sink"),
                    Patch(facecolor=C_REG, label="trainable sink"),
                    Patch(facecolor=C_SEP, label="distributed sink"),
                    Patch(facecolor=C_CT, label="content")],
           loc="upper center", bbox_to_anchor=(_cx, 1.0), bbox_transform=fig.transFigure,
           ncol=4, frameon=False, fontsize=10.8, handlelength=1.1, handleheight=0.9,
           columnspacing=1.2, handletextpad=0.35, borderpad=0.0, borderaxespad=0.0)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
