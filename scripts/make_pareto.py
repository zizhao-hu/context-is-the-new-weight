#!/usr/bin/env python3
"""Pareto view of Table 2's CPT block: WikiText streaming ppl at the deploy budget
against held-out HotpotQA containment accuracy, both under matched (stream) deploy.
Numbers are the tab:toydeploy CPT cells. The frontier is computed over the CPT
designs; the base model is the no-training reference, not a design.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
figstyle.apply()

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/pareto.png")

# label, family, stream ppl (>Cmax), stream task acc
PTS = [
    ("A", "A", 10.08, 0.36), ("A+p0 tok", "A", 11.10, 0.39),
    ("A+p0 scl", "A", 12.33, 0.17), ("A+p0 pre", "A", 11.02, 0.36),
    ("B", "B", 10.02, 0.22), ("B+p0 tok", "B", 9.70, 0.33),
    ("B+p0 scl", "B", 9.86, 0.35), ("B+p0 pre", "B", 9.75, 0.36),
    ("B+sl tok", "B", 9.70, 0.37), ("B+sl pre", "B", 9.82, 0.33),
    ("C", "CD", 9.70, 0.37), ("D", "CD", 9.96, 0.07),
    ("E", "E", 11.50, 0.12), ("E+p0 tok", "E", 10.63, 0.38),
    ("E+p0 pre", "E", 10.59, 0.37), ("E+p0 scl", "E", 10.80, 0.41),
    ("E+sl tok", "E", 10.61, 0.39), ("E+sl pre", "E", 10.54, 0.37),
    ("F", "F", 11.12, 0.33),
]
BASE = (9.52, 0.50)

COLOR = {"A": "0.45", "B": figstyle.C_P0, "CD": figstyle.C_SEP, "E": figstyle.C_REG, "F": "0.2"}
MARK = {"A": "o", "B": "s", "CD": "^", "E": "o", "F": "D"}

front = [p for p in PTS
         if not any(q[2] <= p[2] and q[3] >= p[3] and (q[2], q[3]) != (p[2], p[3])
                    for q in PTS)]
front.sort(key=lambda p: p[2])

fig, ax = plt.subplots(figsize=(figstyle.COL, 1.72))
fx = [p[2] for p in front] + [12.6]
fy = [front[0][3]] + [p[3] for p in front]
ax.step([front[0][2]] + fx[:-1] + [12.6], fy + [fy[-1]], where="post",
        color="0.72", lw=1.0, zorder=1)
for name, fam, x, y in PTS:
    ax.scatter(x, y, s=15 if fam != "E" else 19, marker=MARK[fam], color=COLOR[fam],
               zorder=3, linewidths=0)
ax.scatter(*BASE, marker="*", s=52, color="0.1", zorder=4)
ax.text(BASE[0] + 0.06, BASE[1] - 0.008, "base (no CPT)", fontsize=figstyle.FS_TICK,
        va="center")
ann = {"B+sl tok": (-0.05, -0.045, "right"), "C": (0.0, 0.035, "center"),
       "E+sl tok": (-0.02, 0.033, "center"), "E+p0 scl": (0.10, 0.008, "left"),
       "E": (0.08, 0.0, "left"), "D": (0.08, 0.0, "left"), "B": (0.08, 0.0, "left"),
       "F": (0.08, 0.0, "left")}
for name, fam, x, y in PTS:
    if name in ann:
        dx, dy, ha = ann[name]
        ax.text(x + dx, y + dy, name, fontsize=figstyle.FS_TICK - 0.5, ha=ha,
                va="center", color=COLOR[fam] if fam != "A" else "0.3")
ax.set_ylim(0.02, 0.56)
ax.set_xlim(9.35, 12.6)
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_xlabel("CPT task: WikiText streaming ppl at the budget ($\\downarrow$)",
              fontsize=figstyle.FS_AXIS, labelpad=1.5)
figstyle.clean(ax)
figstyle.yname(ax, "held-out acc. ($\\uparrow$)", pad=0.085)
hs = [plt.Line2D([], [], ls="", marker=MARK[f], color=COLOR[f], ms=4, label=l)
      for f, l in (("A", "A family"), ("B", "B family"), ("CD", "C / D"), ("E", "E family"), ("F", "F"))]
ax.legend(handles=hs, loc="lower right", ncol=2, fontsize=figstyle.FS_LEGEND - 0.5,
          handletextpad=0.15, columnspacing=0.7, borderpad=0.1, labelspacing=0.25)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
