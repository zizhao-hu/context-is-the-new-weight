#!/usr/bin/env python3
"""Where a distributed sink sits, and what happens to it as it ages, side by side.

Left: for each deep query, the sinks inside its window are ranked by nearness and each takes a
share of that query's sink mass. Answers which sink a query picks.

Right: each sink column followed from the step it enters a window to the step it is evicted,
averaged over sinks. Answers whether a given column gains or loses as it ages. Pooling every
query-sink pair instead would confound distance with sink identity.

Data: rankprof_{full,swa,sswa}.json from sink_rank2.py and sinklife64_*.npz from sink_life.py
(Llama-3.2-3B CPT, W=1024, held-out WikiText).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
figstyle.apply()

TR = os.path.expanduser("~/.claude/jobs/f25a34dc/tmp/")
TL = os.path.expanduser("~/.claude/jobs/f25a34dc/tmp/slife/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/sink_pair.png")

MODELS = [("full", "A. full causal", "#4C72B0", "o"),
          ("swa", "B. SWA", "#DD5B45", "s"),
          ("sswa", "E. T-SWA", "#55A868", "^")]
RANKS = list(range(1, 9))
EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1025]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(figstyle.FULL, 1.95),
                             gridspec_kw={"wspace": 0.16})

for key, label, col, mk in MODELS:
    d = {int(k): v for k, v in json.load(open(TR + "rankprof_%s.json" % key)).items()}
    m = [d.get(r, {}).get("mean", np.nan) for r in RANKS]
    e = [d.get(r, {}).get("sem", np.nan) for r in RANKS]
    ax.errorbar(RANKS, m, yerr=e, color=col, marker=mk, ms=3.0, lw=1.1,
                elinewidth=0.7, capsize=1.2, capthick=0.7, label=label)
ax.set_xticks(RANKS)
ax.set_xlabel("sink rank in window (1 $=$ nearest)", fontsize=figstyle.FS_AXIS)
ax.set_yticks([0.0, 0.2, 0.4])
figstyle.clean(ax)
ax.set_title("which sink a query uses", fontsize=figstyle.FS_TITLE - 1.0, pad=3,
             loc="left", color="0.25")

for key, label, col, mk in MODELS:
    C = np.load(TL + "sinklife64_%s.npz" % key)["curves"].astype(float)
    cx, cy, ce = [], [], []
    for i in range(len(EDGES) - 1):
        v = C[:, EDGES[i] - 1:EDGES[i + 1] - 1].mean(1)
        cx.append(np.sqrt(EDGES[i] * EDGES[i + 1]))
        cy.append(v.mean()); ce.append(v.std() / np.sqrt(len(v)))
    bx.errorbar(cx, cy, yerr=ce, color=col, marker=mk, ms=3.0, lw=1.1,
                elinewidth=0.7, capsize=1.2, capthick=0.7, label="%s (n=%d)" % (label, len(C)))
bx.set_xscale("log")
bx.set_xlabel("token age in the window", fontsize=figstyle.FS_AXIS)
figstyle.clean(bx)
bx.legend(frameon=False, fontsize=figstyle.FS_LEGEND, handlelength=1.3, labelspacing=0.22,
          borderpad=0.1, loc="upper right")
bx.set_title("what happens as it ages", fontsize=figstyle.FS_TITLE - 1.0, pad=3,
             loc="left", color="0.25")

figstyle.yname(ax, "share of sink mass", pad=0.085)
figstyle.yname(bx, "mass received", pad=0.085)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
