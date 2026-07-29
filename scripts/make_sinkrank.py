#!/usr/bin/env python3
"""Where in the window does a distributed sink sit?

For every deep query, the sink columns inside its window are ranked by distance (1 = nearest to
the query, larger = further back toward the trailing edge) and each one's share of that query's
total sink mass is recorded. Bars are means over queries with +-1 SEM.

Full-causal training concentrates sink mass on the nearest column and decays away from it.
Window-trained models do not: SWA peaks one step back and keeps substantial mass deep into the
window, and T-SWA is nearly flat, spreading across whatever separators are present.

Data: rankprof_{full,swa,sswa}.json from sink_rank2.py (Llama-3.2-3B CPT, W=1024, 5x-uniform
criterion, held-out WikiText).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import figstyle
figstyle.apply()


T = os.path.expanduser("~/.claude/jobs/f25a34dc/tmp/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/sink_rank.png")

MODELS = [("full", "A. full causal", "#4C72B0"),
          ("swa", "B. SWA", "#DD5B45"),
          ("sswa", "E. T-SWA", "#55A868")]
RANKS = list(range(1, 9))

data = {}
for key, _, _ in MODELS:
    d = json.load(open(T + "rankprof_%s.json" % key))
    data[key] = {int(k): v for k, v in d.items()}

fig, ax = plt.subplots(figsize=(3.34, 2.25))          # single column
x = np.arange(len(RANKS))
w = 0.27
for i, (key, label, col) in enumerate(MODELS):
    m = [data[key].get(r, {}).get("mean", np.nan) for r in RANKS]
    e = [data[key].get(r, {}).get("sem", np.nan) for r in RANKS]
    ax.bar(x + (i - 1) * w, m, w, yerr=e, color=col, label=label,
           error_kw=dict(elinewidth=0.7, capsize=1.2, capthick=0.7, ecolor="0.25"))
ax.set_xticks(x); ax.set_xticklabels([str(r) for r in RANKS], fontsize=figstyle.FS_AXIS)
ax.set_xlabel("sink rank in window (1 $=$ nearest)", fontsize=figstyle.FS_AXIS)
ax.tick_params(labelsize=figstyle.FS_TICK, length=3.0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=figstyle.FS_LEGEND, handlelength=0.9, handleheight=0.8,
          labelspacing=0.25, borderpad=0.1, loc="upper right")
figstyle.yname(ax, "share of sink mass")
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
for key, label, _ in MODELS:
    row = " ".join("%.3f" % data[key].get(r, {}).get("mean", float("nan")) for r in RANKS)
    print("  %-16s %s" % (label, row))
