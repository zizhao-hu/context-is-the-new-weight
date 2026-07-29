#!/usr/bin/env python3
"""Does a distributed sink's mass depend on how far back it sits?

Every (query, sink column) pair inside the window contributes one point: the token distance
q-k and the attention mass column k receives from query q. Points are averaged in log-spaced
distance bins, +-1 SEM, with the rank correlation over the raw pairs in the legend.

Full-causal training grades the mass by recency, nearest separator first. Window training
inverts that: SWA loads the far edge of its window, the tokens about to be evicted. T-SWA is
nearly flat, with a shallow mid-window hump.

Data: sinkdist_{full,swa,sswa}.npz from sink_dist.py (Llama-3.2-3B CPT, W=1024, C=2048,
5x-uniform criterion, held-out WikiText).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import figstyle
figstyle.apply()

from scipy.stats import spearmanr

T = os.path.expanduser("~/.claude/jobs/f25a34dc/tmp/sdist/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/sink_dist.png")

MODELS = [("full", "A. full causal", "#4C72B0", "o"),
          ("swa", "B. SWA", "#DD5B45", "s"),
          ("sswa", "E. T-SWA", "#55A868", "^")]
EDGES = np.unique(np.round(np.logspace(0, np.log10(1024), 13)).astype(int))

fig, ax = plt.subplots(figsize=(3.34, 2.25))
for key, label, col, mk in MODELS:
    d = np.load(T + "sinkdist_%s.npz" % key)
    x, y = d["dist"].astype(float), d["mass"].astype(float)
    cx, cy, ce = [], [], []
    for i in range(len(EDGES) - 1):
        s = (x >= EDGES[i]) & (x < EDGES[i + 1])
        if s.sum() < 6:
            continue
        cx.append(np.sqrt(EDGES[i] * EDGES[i + 1]))
        cy.append(y[s].mean())
        ce.append(y[s].std() / np.sqrt(s.sum()))
    rho = spearmanr(x, y).correlation
    ax.errorbar(cx, cy, yerr=ce, color=col, marker=mk, ms=3.2, lw=1.0,
                elinewidth=0.7, capsize=1.2, capthick=0.7,
                label=r"%s  $\rho{=}%+.2f$" % (label, rho))
ax.set_xscale("log")
ax.set_xlabel("distance back from query (tokens)", fontsize=figstyle.FS_AXIS)
ax.tick_params(labelsize=figstyle.FS_TICK, length=3.0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=figstyle.FS_LEGEND, handlelength=1.4, labelspacing=0.25,
          borderpad=0.1, loc="upper left", ncol=1, bbox_to_anchor=(0.085, 1.0))   # clear of the inside axis name
figstyle.yname(ax, "sink mass per column")
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
