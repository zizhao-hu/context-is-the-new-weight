#!/usr/bin/env python3
"""Where in the window does a distributed sink sit?

For every deep query, the sink columns inside its window are ranked by distance (1 = nearest to
the query, larger = further back toward the trailing edge) and each one's share of that query's
total sink mass is recorded. Bars are means over queries with +-1 SEM.

Full-causal training concentrates sink mass on the nearest column and decays away from it.
Window-trained models do not: SWA peaks one step back and keeps substantial mass deep into the
window, and S-SWA is nearly flat, spreading across whatever separators are present.

Data: rankprof_{full,swa,sswa}.json from sink_rank2.py (Llama-3.2-3B CPT, W=1024, 5x-uniform
criterion, held-out WikiText).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = os.path.expanduser("~/.claude/jobs/f25a34dc/tmp/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/sink_rank.png")

MODELS = [("full", "A. full causal", "#4C72B0"),
          ("swa", "B. SWA", "#DD5B45"),
          ("sswa", "E. S-SWA", "#55A868")]
RANKS = list(range(1, 9))

data = {}
for key, _, _ in MODELS:
    d = json.load(open(T + "rankprof_%s.json" % key))
    data[key] = {int(k): v for k, v in d.items()}

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9})
fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.9), sharey=True)
x = np.arange(len(RANKS))
SHAPE = {"full": "concentrated on the nearest",
         "swa": "spread across the window",
         "sswa": "flat"}
for ax, (key, label, col) in zip(axes, MODELS):
    m = [data[key].get(r, {}).get("mean", np.nan) for r in RANKS]
    e = [data[key].get(r, {}).get("sem", np.nan) for r in RANKS]
    ax.bar(x, m, 0.74, yerr=e, color=col,
           error_kw=dict(elinewidth=0.9, capsize=2.0, capthick=0.9, ecolor="0.2"))
    ax.set_title("%s\n%s" % (label, SHAPE[key]), fontsize=10.5, pad=6, color="0.2")
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in RANKS], fontsize=9)
    ax.tick_params(labelsize=9.5, length=3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("share of sink mass", fontsize=10.5)
fig.supxlabel("distributed sink rank inside the window   "
              r"($1 =$ nearest to the query $\rightarrow$ furthest back)", fontsize=10.5, y=-0.02)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
for key, label, _ in MODELS:
    row = " ".join("%.3f" % data[key].get(r, {}).get("mean", float("nan")) for r in RANKS)
    print("  %-16s %s" % (label, row))
