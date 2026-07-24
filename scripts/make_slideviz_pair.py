#!/usr/bin/env python3
"""Paper Fig 5 rework: real-sequence sliding-window staircase, SWA vs S-SWA.

Both models are Qwen2.5-0.5B CPT at W=32 on the same data/steps; the only
difference is the loss rule (all rows vs full-window rows only). Same real
WikiText sequence, same style as the original slideviz_seq figure:
each staircase row is the W-token window at its true position, cells shaded by
the attention the predicted token (blue box) pays them; green bars on top are
the per-token autoregressive p(token | window).

Data: slideviz.npz (SWA: trained_win/trained_prob_win) and slideviz_sswa.npz.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import matplotlib.cm as cm

D1 = "/Users/zizhaohu/.claude/jobs/f25a34dc/tmp/slideviz.npz"
D2 = "/Users/zizhaohu/.claude/jobs/f25a34dc/tmp/slideviz_sswa.npz"
OUT = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/figures/slideviz_seq.png"

d1 = np.load(D1, allow_pickle=True)
d2 = np.load(D2, allow_pickle=True)
toks = [str(t) for t in d1["tokens"]]
W = int(d1["W"]); L = int(d1["L"])
assert list(map(str, d2["tokens"])) == toks, "sequences differ between captures"

PANELS = [("SWA CPT (loss on all rows)", d1["trained_win"], d1["trained_prob_win"]),
          ("S-SWA CPT (loss on full-window rows only)", d2["trained_win"], d2["trained_prob_win"])]

ROWQ = list(range(W + 1, L - 1, 3))          # query positions with a full window
CMAP = cm.get_cmap("Reds")
NORM = Normalize(vmin=0.0, vmax=0.35)

fig, axes = plt.subplots(2, 1, figsize=(13.2, 9.6))
for ax, (title, att, prob) in zip(axes, PANELS):
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=26)
    # green prob bars along the top
    for t in range(1, L):
        ax.add_patch(Rectangle((t, len(ROWQ) + 0.35), 0.9, 2.6 * float(prob[t]),
                               fc="#2e8b57", ec="none"))
    mp = float(np.mean(prob[W:]))
    ax.text(L + 0.6, len(ROWQ) + 0.9, "mean $p{\\approx}%.2f$" % mp,
            fontsize=12.5, fontweight="bold", color="#2e8b57", va="bottom")
    # staircase
    for r, q in enumerate(ROWQ):
        y = len(ROWQ) - 1 - r
        for k in range(q - W, q):
            ax.add_patch(Rectangle((k, y), 1, 0.92, fc=CMAP(NORM(float(att[q, k]))),
                                   ec="white", lw=0.3))
        ax.add_patch(Rectangle((q, y), 1, 0.92, fill=False, ec="#1f77b4", lw=1.6))
    ax.set_xlim(0, L + 7); ax.set_ylim(-0.3, len(ROWQ) + 3.4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    # token strip along the bottom
    for t in range(L):
        ax.text(t + 0.5, -0.55, toks[t].replace(" ", "·"), fontsize=4.6,
                rotation=90, ha="center", va="top", color="0.25")
    ax.set_ylim(-3.3, len(ROWQ) + 3.4)

fig.subplots_adjust(left=0.015, right=0.995, top=0.96, bottom=0.01, hspace=0.10)
sm = cm.ScalarMappable(norm=NORM, cmap=CMAP)
cax = fig.add_axes([0.86, 0.475, 0.11, 0.014])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("attention received", fontsize=10)
cb.ax.tick_params(labelsize=8.5)
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("wrote", OUT)
