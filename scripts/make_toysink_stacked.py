#!/usr/bin/env python3
"""Finding-1 support figure (toy_sink.png) in the a_sinkbars STACKED style.

Left: from-scratch BPE toy GPT (sink-head attention, W=64) -- full causal vs SWA.
Right: continued pretraining Llama-3.2-3B (mean head, W=512, n=12) -- base / full / SWA.
Stacked classes per model: p0 sink + common-token sink + content (remainder to 1),
white value labels, +-1 SEM caps at segment edges, top legend (make_a_sinkbars.py style).
Numbers from scripts/make_tri_2panel.py (measured; toy sink-head, CPT mean-head).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/figures/toy_sink.png"

# (p0, p0sem, common, commonsem)
TOY = [("full", 0.5770, 0.0190, 0.1190, 0.0100),
       ("SWA",  0.0000, 0.0000, 0.2280, 0.0410)]
CPT = [("base", 0.5855, 0.0031, 0.0479, 0.0011),
       ("full", 0.6404, 0.0046, 0.0537, 0.0018),
       ("SWA",  0.1630, 0.0267, 0.5017, 0.0311)]

C_P0, C_SEP, C_CT = "#4C72B0", "#55A868", "#D3D3D3"
plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.65), sharey=True,
                         gridspec_kw={"width_ratios": [2, 3]})

T_L = "pretrained from scratch"
T_R = "continued pretraining (Llama-3.2-3B)"
for ax, rows, ttl in ((axes[0], TOY, T_L), (axes[1], CPT, T_R)):
    x = np.arange(len(rows))
    labs = [r[0] for r in rows]
    p0 = np.array([r[1] for r in rows]); p0s = np.array([r[2] for r in rows])
    cm = np.array([r[3] for r in rows]); cms = np.array([r[4] for r in rows])
    ct = 1.0 - p0 - cm
    w = 0.80
    ax.bar(x, p0, w, color=C_P0, edgecolor="white", linewidth=0.6)
    ax.bar(x, cm, w, bottom=p0, color=C_SEP, edgecolor="white", linewidth=0.6)
    ax.bar(x, ct, w, bottom=p0 + cm, color=C_CT, edgecolor="white", linewidth=0.6)
    edges = np.stack([p0, p0 + cm], axis=1)
    sv = np.stack([p0s, cms], axis=1)
    for j in range(2):
        ax.errorbar(x, edges[:, j], yerr=sv[:, j], fmt="none", ecolor="0.15",
                    elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=5)
    for xx, a_, b_ in zip(x, p0, cm):
        for yy, v in ((a_ / 2, a_), (a_ + b_ / 2, b_)):
            if v >= 0.02:
                ax.text(xx, yy, "%.2f" % v, ha="center", va="center", fontsize=10.6,
                        color="white", fontweight="bold")
    ax.set_ylim(0, 1); ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=11.6)
    ax.set_xlim(-0.60, len(rows) - 0.40)
    ax.set_title(ttl, fontsize=11.6, fontweight="bold", pad=3)
    ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.tick_params(length=3.5, labelsize=10.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("attention mass", fontsize=11.6)

plt.tight_layout(rect=(0, 0, 1, 0.90))
_cx = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
fig.legend(handles=[Patch(facecolor=C_P0, label="p0 sink"),
                    Patch(facecolor=C_SEP, label="distributed sink"),
                    Patch(facecolor=C_CT, label="content")],
           loc="upper center", bbox_to_anchor=(_cx, 1.0), bbox_transform=fig.transFigure,
           ncol=3, frameon=False, fontsize=11.0, handlelength=1.15, handleheight=0.9,
           columnspacing=1.6, handletextpad=0.4, borderpad=0.0, borderaxespad=0.0)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
