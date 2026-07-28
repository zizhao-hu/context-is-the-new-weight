#!/usr/bin/env python3
"""Finding-2 figure: trainable sinks offload both structural sinks, on BOTH bases.

Left: full-causal base + sink designs (adecomp_5151160.log).
Right: SWA base + sink designs (fulldecomp_5166496.log).
BPE toy, W=128, C=256, within max context, all-heads means, n=32, +-1 SEM.
Stacked classes: p0 sink + trainable sink + distributed sink + content (sum to 1).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/sink_absorb.png")

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
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "hatch.linewidth": 0.35, "hatch.color": "0.15"})


def pale(c, f=0.45):
    """same hue, lighter: distinguishes the SWA bar without a texture"""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

# four designs, two bars per group: full-causal base and SWA base side by side
DESIGNS = ["base", "+p0 token", "+p0 scalar", "+p0 prefix"]
def col(fam, i):
    r = fam[i]
    return np.array([r[1], r[2], r[3], r[4], r[5], r[6]], dtype=float)

fig, ax = plt.subplots(figsize=(3.34, 2.05))            # single column
x = np.arange(len(DESIGNS))
w, dx = 0.36, 0.20
for k, fam in enumerate((FULLFAM, SWAFAM)):
    tint = (lambda c: c) if k == 0 else pale
    hat = None if k == 0 else "////"
    p0 = np.array([fam[i][1] for i in range(4)]); p0s = np.array([fam[i][2] for i in range(4)])
    rg = np.array([fam[i][3] for i in range(4)]); rgs = np.array([fam[i][4] for i in range(4)])
    sp = np.array([fam[i][5] for i in range(4)]); sps = np.array([fam[i][6] for i in range(4)])
    ct = 1.0 - p0 - rg - sp
    xx = x + (dx if k else -dx)
    ax.bar(xx, rg, w, color=tint(C_REG), edgecolor="black", linewidth=0.6, hatch=hat, zorder=3)
    ax.bar(xx, p0, w, bottom=rg, color=tint(C_P0), edgecolor="black", linewidth=0.6, hatch=hat, zorder=3)
    ax.bar(xx, sp, w, bottom=rg + p0, color=tint(C_SEP), edgecolor="black", linewidth=0.6, hatch=hat, zorder=3)
    ax.bar(xx, ct, w, bottom=rg + p0 + sp, color=tint(C_CT), edgecolor="black", linewidth=0.6, hatch=hat, zorder=3)
    edges = np.stack([rg, rg + p0, rg + p0 + sp], axis=1)
    sv = np.stack([rgs, p0s, sps], axis=1)
    for j in range(3):
        ax.errorbar(xx, edges[:, j], yerr=sv[:, j], fmt="none", ecolor="0.15",
                    elinewidth=0.7, capsize=1.2, capthick=0.7, zorder=6)
ax.set_ylim(0, 1.06); ax.set_xticks(x)
ax.set_xticklabels(DESIGNS, fontsize=8)
for xi in x:
    ax.text(xi - dx, 1.015, "F", fontsize=6.5, ha="center", va="bottom", color="0.35")
    ax.text(xi + dx, 1.015, "S", fontsize=6.5, ha="center", va="bottom", color="0.35")
ax.set_xlim(-0.55, len(DESIGNS) - 0.45)
ax.set_ylabel("attention mass", fontsize=8.5)
ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.tick_params(labelsize=8, length=2.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
h = [Patch(facecolor=C_REG, edgecolor="black", lw=0.6, label="trainable sink"),
     Patch(facecolor=C_P0, edgecolor="black", lw=0.6, label="p0 sink"),
     Patch(facecolor=C_SEP, edgecolor="black", lw=0.6, label="distributed"),
     Patch(facecolor=C_CT, edgecolor="black", lw=0.6, label="content"),
     Patch(facecolor="0.55", edgecolor="black", lw=0.6, label="F: full causal"),
     Patch(facecolor=pale("0.55"), edgecolor="black", lw=0.6, hatch="////", label="S: SWA")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False,
           fontsize=7.0, handlelength=0.9, handleheight=0.8, columnspacing=0.8,
           handletextpad=0.3, labelspacing=0.2)
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
