#!/usr/bin/env python3
"""One sink's mass across its own lifetime in the window.

Replaces the pooled distance plot, which was confounded: pooling every (query, sink) pair mixes
sink columns of different strengths, so the far bins could be carried by a few strong columns
rather than by any column getting stronger. The old criterion made it worse by scoring `recv`
over a truncated window near the end of the sequence, which over-selected late positions.

Here each sink column is followed from the moment it enters a query's window to the moment it
slides out, giving one curve per sink, and the curves are averaged. Context is 3W/2 so every
sink's whole lifetime is seen by queries that are themselves deep.

Data: sinklife64_{full,swa,sswa}.npz from sink_life.py (Llama-3.2-3B CPT, W=1024, 64 sequences).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
figstyle.apply()

T = os.path.expanduser("~/.claude/jobs/f25a34dc/tmp/slife/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/sink_life.png")

MODELS = [("full", "A. full causal", "#4C72B0", "o"),
          ("swa", "B. SWA", "#DD5B45", "s"),
          ("sswa", "E. T-SWA", "#55A868", "^")]
EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1025]

fig, ax = plt.subplots(figsize=(figstyle.COL, 1.86))
for key, label, col, mk in MODELS:
    C = np.load(T + "sinklife64_%s.npz" % key)["curves"].astype(float)
    cx, cy, ce = [], [], []
    for i in range(len(EDGES) - 1):
        sl = slice(EDGES[i] - 1, EDGES[i + 1] - 1)
        v = C[:, sl].mean(1)
        cx.append(np.sqrt(EDGES[i] * EDGES[i + 1]))
        cy.append(v.mean()); ce.append(v.std() / np.sqrt(len(v)))
    ax.errorbar(cx, cy, yerr=ce, color=col, marker=mk, ms=3.0, lw=1.1,
                elinewidth=0.7, capsize=1.2, capthick=0.7, label="%s  (n=%d)" % (label, len(C)))
ax.set_xscale("log")
ax.set_xlabel("token age in the window", fontsize=figstyle.FS_AXIS)
figstyle.clean(ax)
ax.legend(frameon=False, fontsize=figstyle.FS_LEGEND, handlelength=1.3, labelspacing=0.22,
          borderpad=0.1, loc="upper right")
figstyle.yname(ax, "mass received")
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
