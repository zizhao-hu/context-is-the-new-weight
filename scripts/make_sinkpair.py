#!/usr/bin/env python3
"""Where a distributed sink sits, and what happens to it as it ages: two panels, one column.

Same construction as the loss-mask ablation figures: a single-column figure carrying two panels
side by side, one shared legend above them, axis names inside the axes, matched tick and label
sizes.

Left: for each deep query the sinks inside its window are ranked by nearness, and each takes a
share of that query's sink mass. Which sink does a query pick?

Right: each sink column followed from the step it enters a window to the step it is evicted,
averaged over sinks. Does a given column gain or lose as it ages? Pooling every query-sink pair
instead would confound distance with sink identity.

Data: rankprof_{full,swa,sswa}.json from sink_rank2.py, sinklife64_*.npz from sink_life.py
(Llama-3.2-3B CPT, W=1024, held-out WikiText).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.51, 1.49),
                             gridspec_kw={"wspace": 0.045})

for key, label, col, mk in MODELS:
    d = {int(k): v for k, v in json.load(open(TR + "rankprof_%s.json" % key)).items()}
    m = [d.get(r, {}).get("mean", np.nan) for r in RANKS]
    e = [d.get(r, {}).get("sem", np.nan) for r in RANKS]
    ax.errorbar(RANKS, m, yerr=e, color=col, marker=mk, ms=2.4, lw=1.0,
                elinewidth=0.6, capsize=1.0, capthick=0.6)
ax.set_xticks([1, 4, 8])
ax.set_yticks([0.0, 0.2, 0.4])
ax.set_xlabel("sink rank", fontsize=figstyle.FS_AXIS)

for key, label, col, mk in MODELS:
    C = np.load(TL + "sinklife64_%s.npz" % key)["curves"].astype(float)
    cx, cy, ce = [], [], []
    for i in range(len(EDGES) - 1):
        v = C[:, EDGES[i] - 1:EDGES[i + 1] - 1].mean(1)
        cx.append(np.sqrt(EDGES[i] * EDGES[i + 1]))
        cy.append(v.mean()); ce.append(v.std() / np.sqrt(len(v)))
    bx.errorbar(cx, cy, yerr=ce, color=col, marker=mk, ms=2.4, lw=1.0,
                elinewidth=0.6, capsize=1.0, capthick=0.6)
bx.set_xscale("log")
bx.set_xticks([1, 10, 100, 1000])
bx.set_xticklabels(["1", "10", "100", "1000"])      # plain numbers, not 10^n
bx.set_ylim(0.0, 0.10)
bx.set_yticks([0.0, 0.1])
bx.set_xlabel("sink distance from q", fontsize=figstyle.FS_AXIS)

for a in (ax, bx):
    figstyle.clean(a)
    a.tick_params(labelsize=figstyle.FS_TICK - 1.0)
figstyle.yname(ax, "share of sink mass", pad=0.115)
bx.yaxis.tick_right()                        # mirrored outward, so the gap stays clear
bx.spines["left"].set_visible(False)
bx.spines["right"].set_visible(True)
bx.spines["right"].set_linewidth(figstyle.LW_AXES)
figstyle.yname(bx, "mass received", pad=0.115, side="right")

fig.legend(handles=[Line2D([], [], color=c, marker=k, ms=2.6, lw=1.0, label=l)
                    for _, l, c, k in MODELS],
           loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
           fontsize=figstyle.FS_LEGEND, handlelength=1.1, handleheight=0.8,
           columnspacing=0.9, handletextpad=0.35, borderpad=0.0)
plt.tight_layout(rect=(0, 0, 1, 0.86))
_pa, _pb = ax.get_position(), bx.get_position()
_xm = (_pa.x1 + _pb.x0) / 2
fig.add_artist(Line2D([_xm, _xm], [_pa.y0, _pa.y1], color="0.75", lw=0.9,
                      ls=":", transform=fig.transFigure))
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
