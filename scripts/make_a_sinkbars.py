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
SHORT = {"a. full causal": "full", "b. sliding window": "SWA",
         "e. cc sliding": "T-SWA"}
xl = [SHORT.get(l, l) for l in labels]

C_P0, C_REG, C_SEP, C_CT = "#4C72B0", "#DD5B45", "#55A868", "#D3D3D3"
fig, axes = plt.subplots(1, 2, figsize=(figstyle.COL, 2.45), sharey=True)   # tall enough to label every segment

T_LO = "within max context"
T_HI = "past max context"
for ax, vals, sv, ttl in ((axes[0], cold, sem_cold, T_LO),
                          (axes[1], deep, sem_deep, T_HI)):
    x = np.arange(len(labels))
    fiw, sep, ct = vals[:, 0], vals[:, 1], vals[:, 2]
    w = 0.80
    ax.bar(x, fiw, w, color=C_P0, edgecolor="black", linewidth=0.6, zorder=3)
    ax.bar(x, sep, w, bottom=fiw, color=C_SEP, edgecolor="black", linewidth=0.6, zorder=3)
    ax.bar(x, ct, w, bottom=fiw + sep, color=C_CT, edgecolor="black", linewidth=0.6, zorder=3)
    edges = np.stack([fiw, fiw + sep], axis=1)
    for j in range(2):
        ax.errorbar(x, edges[:, j], yerr=sv[:, j], fmt="none", ecolor="0.15",
                    elinewidth=0.9, capsize=1.6, capthick=0.8, zorder=5)
    # label the three sink classes; content is the remainder to 1 and needs no number
    for xx, a_, b_ in zip(x, fiw, sep):
        for yy, v in ((a_ / 2, a_), (a_ + b_ / 2, b_)):
            if v >= 0.01:
                ax.text(xx, yy, "%.2f" % v, ha="center", va="center", fontsize=figstyle.FS_CHIP,
                        color="0.1", fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="0.45", lw=0.7, alpha=0.92))
    ax.set_ylim(0, 1); ax.set_xticks(x)
    ax.set_xticklabels(xl, fontsize=figstyle.FS_TICK)
    ax.set_xlim(-0.60, len(labels) - 0.40)
    ax.set_title(ttl, fontsize=figstyle.FS_TICK, fontweight="bold", pad=3)
    ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.tick_params(length=3.5, labelsize=figstyle.FS_TICK)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.legend(handles=[Patch(facecolor=C_P0, edgecolor="black", lw=0.6, label="first-in-window sink"),
                    Patch(facecolor=C_SEP, edgecolor="black", lw=0.6, label="distributed sink"),
                    Patch(facecolor=C_CT, edgecolor="black", lw=0.6, label="content")],
           loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False,
           fontsize=figstyle.FS_LEGEND, handlelength=0.9, handleheight=0.8, columnspacing=0.8,
           handletextpad=0.3, labelspacing=0.2)
figstyle.yname(axes[0], "attention mass", also=(axes[1],), pad=0.105)
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
