#!/usr/bin/env python3
"""Paper Fig 5: S-SWA minus SWA on a real sequence, one panel.

Both models are Qwen2.5-0.5B CPT at W=32 on the same data/steps; the only
difference is the loss rule (all rows vs full-window rows only). This figure
shows the difference between them directly: staircase cells are shaded by the
change in attention the predicted token (blue box) pays each key, and the bars
on top are the change in p(token | window). Red means S-SWA attends more or is
more confident, blue means less.

Data: slideviz.npz (SWA: trained_win/trained_prob_win) and slideviz_sswa.npz.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

D1 = "/Users/zizhaohu/.claude/jobs/f25a34dc/tmp/slideviz.npz"
D2 = "/Users/zizhaohu/.claude/jobs/f25a34dc/tmp/slideviz_sswa.npz"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/slideviz_seq.png")

d1 = np.load(D1, allow_pickle=True)
d2 = np.load(D2, allow_pickle=True)
toks = [str(t) for t in d1["tokens"]]
W = int(d1["W"]); L = int(d1["L"])
assert list(map(str, d2["tokens"])) == toks, "sequences differ between captures"

ATT1, P1 = d1["trained_win"], d1["trained_prob_win"]
ATT2, P2 = d2["trained_win"], d2["trained_prob_win"]
DATT = ATT2 - ATT1                                   # S-SWA minus SWA
DPROB = np.asarray(P2, dtype=float) - np.asarray(P1, dtype=float)

# first separator-sink column: strongest received column among full-window queries (SWA capture)
_recv = np.array([ATT1[k + 1:min(k + 1 + W, L), k].mean() if k + 1 < L else 0 for k in range(L)])
_recv[0] = 0
_strong = np.where(_recv > 0.5 * _recv.max())[0]
_ok = [k for k in _strong if k <= L - W - 6]
SINK = int(_ok[0]) if _ok else int(np.argmax(_recv))   # earliest strong sink column that fits a window
X0 = max(0, SINK - 7)                        # view starts 7 tokens left of the sink column
ROWQ = [X0 + W - 1 + 5 * r for r in range(5)]           # 5 rows, 5 tokens apart
ROWQ = [q for q in ROWQ if q < L]
X1 = min(L, ROWQ[-1] + 2)                    # view ends at the last predicted token

# symmetric scale from the cells actually drawn
_cells = [abs(float(DATT[q, k])) for q in ROWQ for k in range(q - W, q)]
AMAX = max(1e-4, float(np.percentile(_cells, 99)))
CMAP = matplotlib.colormaps["RdBu_r"]
NORM = TwoSlopeNorm(vmin=-AMAX, vcenter=0.0, vmax=AMAX)
PMAX = max(1e-3, float(np.abs(DPROB[X0:X1]).max()))
BARH = 1.5                                    # rows of vertical space for the bar strip

fig, ax = plt.subplots(figsize=(13.2, 5.6))
NR = len(ROWQ)
base = NR + 0.45 + BARH                       # zero line of the probability bars

# probability-difference bars, signed around their own zero line
ax.plot([X0 - 0.2, X1 - 0.8], [base, base], color="0.55", lw=0.8, zorder=2)
for t in range(X0, X1):
    v = float(DPROB[t])
    h = BARH * v / PMAX
    ax.add_patch(Rectangle((t + 0.05, base), 0.9, h, zorder=3,
                           fc=("#b2182b" if v >= 0 else "#2166ac"), ec="none"))
ax.text(X1 - 0.6, base, r"  $\Delta p$", fontsize=12, va="center", color="0.25")
ax.text(X1 - 0.6, base + BARH * 0.75, "  %+.2f" % PMAX, fontsize=9.5, va="center", color="#b2182b")
ax.text(X1 - 0.6, base - BARH * 0.75, "  %+.2f" % -PMAX, fontsize=9.5, va="center", color="#2166ac")

# staircase of attention differences
for r, q in enumerate(ROWQ):
    y = NR - 1 - r
    for k in range(q - W, q):
        ax.add_patch(Rectangle((k, y), 1, 0.92, fc=CMAP(NORM(float(DATT[q, k]))),
                               ec="white", lw=0.3))
    ax.add_patch(Rectangle((q, y), 1, 0.92, fill=False, ec="#1f77b4", lw=1.6))

ax.set_xlim(X0 - 0.5, X1 + 7)
ax.set_ylim(-3.3, base + BARH + 0.5)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
for t in range(X0, X1):
    ax.text(t + 0.5, -0.55, toks[t].replace(" ", "·"), fontsize=8.5,
            rotation=90, ha="center", va="top", color="0.25")

fig.subplots_adjust(left=0.015, right=0.995, top=0.97, bottom=0.02)
sm = cm.ScalarMappable(norm=NORM, cmap=CMAP)
cax = fig.add_axes([0.855, 0.60, 0.11, 0.022])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label(r"$\Delta$ attention (S-SWA $-$ SWA)", fontsize=10)
cb.set_ticks([-AMAX, 0, AMAX])
cb.ax.set_xticklabels(["%+.2f" % -AMAX, "0", "%+.2f" % AMAX])
cb.ax.tick_params(labelsize=8.5)
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("wrote", OUT, "| amax %.4f pmax %.4f" % (AMAX, PMAX))
