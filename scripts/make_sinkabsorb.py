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
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})

# one panel, both bases in a row; the first bar of each base is its own training with no sink
ROWS = [("full", ) + FULLFAM[0][1:]] + [("f " + r[0].replace("+sink ", "+"),) + r[1:] for r in FULLFAM[1:]]      + [("SWA", ) + SWAFAM[0][1:]] + [("s " + r[0].replace("+sink ", "+"),) + r[1:] for r in SWAFAM[1:]]
labs = [r[0] for r in ROWS]
p0 = np.array([r[1] for r in ROWS]); p0s = np.array([r[2] for r in ROWS])
rg = np.array([r[3] for r in ROWS]); rgs = np.array([r[4] for r in ROWS])
sp = np.array([r[5] for r in ROWS]); sps = np.array([r[6] for r in ROWS])
ct = 1.0 - p0 - rg - sp
x = np.arange(len(ROWS))

fig, ax = plt.subplots(figsize=(3.34, 2.5))            # single column
w = 0.80
ax.bar(x, rg, w, color=C_REG, edgecolor="white", linewidth=0.5)
ax.bar(x, p0, w, bottom=rg, color=C_P0, edgecolor="white", linewidth=0.5)
ax.bar(x, sp, w, bottom=rg + p0, color=C_SEP, edgecolor="white", linewidth=0.5)
ax.bar(x, ct, w, bottom=rg + p0 + sp, color=C_CT, edgecolor="white", linewidth=0.5)
edges = np.stack([rg, rg + p0, rg + p0 + sp], axis=1)
sv = np.stack([rgs, p0s, sps], axis=1)
for j in range(3):
    ax.errorbar(x, edges[:, j], yerr=sv[:, j], fmt="none", ecolor="0.15",
                elinewidth=0.7, capsize=1.2, capthick=0.7, zorder=5)
ax.axvline(3.5, color="0.35", lw=0.8, ls=":")          # full-causal base | SWA base
ax.set_ylim(0, 1); ax.set_xticks(x)
ax.set_xticklabels(labs, fontsize=7, rotation=38, ha="right")
ax.set_xlim(-0.6, len(ROWS) - 0.4)
ax.set_ylabel("attention mass", fontsize=8.5)
ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.tick_params(labelsize=8, length=2.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.legend(handles=[Patch(facecolor=C_REG, label="trainable sink"),
                    Patch(facecolor=C_P0, label="p0 sink"),
                    Patch(facecolor=C_SEP, label="distributed sink"),
                    Patch(facecolor=C_CT, label="content")],
           loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False,
           fontsize=7.4, handlelength=0.9, handleheight=0.8, columnspacing=0.9,
           handletextpad=0.35, labelspacing=0.2)
plt.tight_layout(rect=(0, 0, 1, 0.88))
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
