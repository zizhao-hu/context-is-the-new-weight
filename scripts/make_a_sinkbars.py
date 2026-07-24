#!/usr/bin/env python3
"""Finding 1 bar chart: the mask decides WHICH sink forms; a trainable sink intercepts it.

Columns: full (causal) | SWA (sliding window) | S-SWA (symmetric SWA: constant context) -- three TRAINING
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
         "e. cc sliding": "S-SWA"}
xl = [SHORT.get(l, l) for l in labels]

C_P0, C_REG, C_SEP, C_CT = "#4C72B0", "#DD5B45", "#55A868", "#D3D3D3"
plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.65), sharey=True)   # tall enough to label every segment

T_LO = "sequence length $<$ max context"
T_HI = "sequence length $>$ max context"
for ax, vals, sv, ttl in ((axes[0], cold, sem_cold, T_LO),
                          (axes[1], deep, sem_deep, T_HI)):
    x = np.arange(len(labels))
    fiw, sep, ct = vals[:, 0], vals[:, 1], vals[:, 2]
    w = 0.80
    ax.bar(x, fiw, w, color=C_P0, edgecolor="white", linewidth=0.6)
    ax.bar(x, sep, w, bottom=fiw, color=C_SEP, edgecolor="white", linewidth=0.6)
    ax.bar(x, ct, w, bottom=fiw + sep, color=C_CT, edgecolor="white", linewidth=0.6)
    edges = np.stack([fiw, fiw + sep], axis=1)
    for j in range(2):
        ax.errorbar(x, edges[:, j], yerr=sv[:, j], fmt="none", ecolor="0.15",
                    elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=5)
    # label the three sink classes; content is the remainder to 1 and needs no number
    for xx, a_, b_ in zip(x, fiw, sep):
        for yy, v in ((a_ / 2, a_), (a_ + b_ / 2, b_)):
            if v >= 0.01:
                ax.text(xx, yy, "%.2f" % v, ha="center", va="center", fontsize=9.8,
                        color="0.1", fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.45", lw=0.8, alpha=0.9))
    ax.set_ylim(0, 1); ax.set_xticks(x)
    ax.set_xticklabels(xl, fontsize=11.6)
    ax.set_xlim(-0.60, len(labels) - 0.40)
    ax.set_title(ttl, fontsize=11.6, fontweight="bold", pad=3)
    ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.tick_params(length=3.5, labelsize=10.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("attention mass", fontsize=11.6)

plt.tight_layout(rect=(0, 0, 1, 0.90))
_cx = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2   # centre over the panels only
fig.legend(handles=[Patch(facecolor=C_P0, label="first-in-window sink"),
                    Patch(facecolor=C_SEP, label="distributed sink"),
                    Patch(facecolor=C_CT, label="content")],
           loc="upper center", bbox_to_anchor=(_cx, 1.0), bbox_transform=fig.transFigure,
           ncol=3, frameon=False, fontsize=11.0, handlelength=1.15, handleheight=0.9,
           columnspacing=1.6, handletextpad=0.4, borderpad=0.0, borderaxespad=0.0)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
