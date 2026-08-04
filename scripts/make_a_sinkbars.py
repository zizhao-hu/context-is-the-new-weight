#!/usr/bin/env python3
"""Finding 1 bar chart: the mask decides WHICH sink forms; a trainable sink intercepts it.

Columns: full (causal) | SWA (sliding window) | T-SWA (truncated SWA: constant context) -- three TRAINING
regimes, no trainable sinks. "pos-0" is the FIRST TOKEN IN THE WINDOW (the oldest visible column,
q-W+1), which under full causal is column 0; scoring absolute column 0 instead would make b/e
exactly 0 at depth by construction.

Data comes from figures/a_sinkdecomp.tsv (written from sinkdecomp_cc.py output); this script
refuses to plot if that file is missing so placeholder numbers can never reach the paper.
Columns: label fiw_c sep_c ct_c  fiw_d sep_d ct_d  fiw_c_sem sep_c_sem fiw_d_sem sep_d_sem
Error bars are +/-1 SEM over the n independent eval segments.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import figstyle
figstyle.apply()

from matplotlib.patches import Patch

DATA = "figures/a_sinkdecomp.tsv"
OUT = "paper/attention-sink/figures/a_sinkbars.png"

if not os.path.exists(DATA):
    sys.exit("missing %s -- run sinkdecomp_cc.py and write its numbers there first" % DATA)

labels, cold, deep, sems = [], [], [], []
for ln in open(DATA, encoding="utf-8"):
    ln = ln.strip()
    if not ln or ln.startswith("#"):
        continue
    f = ln.split("\t")
    labels.append(f[0])
    cold.append([float(x) for x in f[1:4]])
    deep.append([float(x) for x in f[4:7]])
    sems.append([float(x) for x in f[7:11]])
cold, deep, sems = np.array(cold), np.array(deep), np.array(sems)
sem_cold, sem_deep = sems[:, 0:2], sems[:, 2:4]

# short x labels -- the long forms don't fit under vertical bars
SHORT = {"a. full causal": "full", "f. truncated full": "T-full", "b. sliding window": "SWA",
         "e. cc sliding": "T-SWA"}
xl = [SHORT.get(l, l) for l in labels]

C_P0, C_REG, C_SEP, C_CT = "#4C72B0", "#DD5B45", "#55A868", "#D3D3D3"

# one axes with two blocks and a dotted divider, exactly like the ablation figures
GAP = 1.35
n = len(labels)
xL = np.arange(n, dtype=float)
xR = xL + n + GAP
fig, ax = plt.subplots(figsize=(figstyle.COL, 1.52))
w = 0.80
for xs, vals, sv in ((xL, cold, sem_cold), (xR, deep, sem_deep)):
    fiw, sep, ct = vals[:, 0], vals[:, 1], vals[:, 2]
    ax.bar(xs, fiw, w, color=C_P0, edgecolor="black", linewidth=0.6, zorder=3)
    ax.bar(xs, sep, w, bottom=fiw, color=C_SEP, edgecolor="black", linewidth=0.6, zorder=3)
    ax.bar(xs, ct, w, bottom=fiw + sep, color=C_CT, edgecolor="black", linewidth=0.6, zorder=3)
    edges = np.stack([fiw, fiw + sep], axis=1)
    for j in range(2):
        ax.errorbar(xs, edges[:, j], yerr=sv[:, j], fmt="none", ecolor="0.15",
                    elinewidth=0.7, capsize=1.2, capthick=0.7, zorder=6)

div = (xL[-1] + xR[0]) / 2.0
ax.axvline(div, color="0.75", lw=0.9, ls=":", zorder=1)
ax.set_ylim(0, 1.30)
ax.set_yticks(np.arange(0, 1.01, 0.25))
ax.set_xticks(np.concatenate([xL, xR]))
ax.set_xticklabels(xl + xl, fontsize=figstyle.FS_TICK - 0.8)
ax.set_xlim(xL[0] - 0.75, xR[-1] + 0.75)
figstyle.clean(ax)
ax.text(np.mean(xL), -0.22, "within max context", ha="center", va="top",
        fontsize=figstyle.FS_AXIS, transform=ax.get_xaxis_transform())
ax.text(np.mean(xR), -0.22, "past max context", ha="center", va="top",
        fontsize=figstyle.FS_AXIS, transform=ax.get_xaxis_transform())

ax.legend(handles=[Patch(facecolor=C_P0, edgecolor="black", lw=0.6, label="first-in-window sink"),
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
