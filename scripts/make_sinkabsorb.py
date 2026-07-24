#!/usr/bin/env python3
"""Finding-2 figure: trainable sinks absorb the parking demand (fig-3 stacked style).

Bars: full causal / +sink token / +sink scalar / +sink prefix (BPE toy, W=128, C=256,
within max context, all-heads means, n=32 segments; adecomp_5151160.log).
Stacked classes: p0 sink + separator sink + trainable sink + content (sum to 1).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/figures/sink_absorb.png"

# (p0, p0sem, reg, regsem, sep, sepsem)
ROWS = [("full",         0.2491, 0.0032, 0.0000, 0.0000, 0.0498, 0.0017),
        ("+sink token",  0.0255, 0.0004, 0.3283, 0.0040, 0.0387, 0.0016),
        ("+sink scalar", 0.0268, 0.0005, 0.3679, 0.0042, 0.0359, 0.0013),
        ("+sink prefix", 0.0152, 0.0003, 0.4322, 0.0038, 0.0294, 0.0013)]

C_P0, C_REG, C_SEP, C_CT = "#4C72B0", "#DD5B45", "#55A868", "#D3D3D3"
plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
fig, ax = plt.subplots(figsize=(5.4, 3.65))

x = np.arange(len(ROWS))
labs = [r[0] for r in ROWS]
p0 = np.array([r[1] for r in ROWS]); p0s = np.array([r[2] for r in ROWS])
rg = np.array([r[3] for r in ROWS]); rgs = np.array([r[4] for r in ROWS])
sp = np.array([r[5] for r in ROWS]); sps = np.array([r[6] for r in ROWS])
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
        if v >= 0.02:
            ax.text(xx, yy, "%.2f" % v, ha="center", va="center", fontsize=10.6,
                    color="white", fontweight="bold")
ax.set_ylim(0, 1); ax.set_xticks(x)
ax.set_xticklabels(labs, fontsize=11.2)
ax.set_xlim(-0.60, len(ROWS) - 0.40)
ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.tick_params(length=3.5, labelsize=10.6)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_ylabel("attention mass", fontsize=11.6)

plt.tight_layout(rect=(0, 0, 1, 0.90))
fig.legend(handles=[Patch(facecolor=C_P0, label="p0 sink"),
                    Patch(facecolor=C_REG, label="trainable sink"),
                    Patch(facecolor=C_SEP, label="separator sink"),
                    Patch(facecolor=C_CT, label="content")],
           loc="upper center", bbox_to_anchor=(0.55, 1.0), bbox_transform=fig.transFigure,
           ncol=4, frameon=False, fontsize=10.4, handlelength=1.1, handleheight=0.9,
           columnspacing=1.0, handletextpad=0.35, borderpad=0.0, borderaxespad=0.0)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
