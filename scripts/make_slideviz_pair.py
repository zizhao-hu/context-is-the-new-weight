#!/usr/bin/env python3
"""Paper Fig 5: what the truncated loss rule does to next-token probability.

Top: per-token change in p(token | window), T-SWA minus SWA, on one real WikiText
sequence, with each token coloured by its class (function word, content word,
punctuation, digit, subword piece).

Bottom: the same difference aggregated by token class over a 2x2 design, two seeds
per loss rule scored on 40 held-out 200-token chunks. The grey band is the
within-condition (same rule, different data order) difference, i.e. the noise floor.

Data: slideviz{,_sswa}_L200h.npz for the top panel, probsweep_{swa,sswa}{0,1}.npz
plus probsweep_toks.npz for the bottom.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import figstyle
figstyle.apply()

from matplotlib.patches import Rectangle, Patch

T = "/Users/zizhaohu/.claude/jobs/f25a34dc/tmp/"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/figures/slideviz_seq.png")

# standard English stop list (NLTK's), so the function-word class is not ad hoc
STOP = set("""i me my myself we our ours ourselves you your yours yourself yourselves he him his
himself she her hers herself it its itself they them their theirs themselves what which who whom
this that these those am is are was were be been being have has had having do does did doing a an
the and but if or because as until while of at by for with about against between into through
during before after above below to from up down in out on off over under again further then once
here there when where why how all any both each few more most other some such no nor not only own
same so than too very s t can will just don should now""".split())


def klass(t):
    s = t.strip()
    if s == "":
        return "punctuation & space"
    if re.fullmatch(r"[^\w\s]+", s):
        return "punctuation & space"
    if s.isdigit():
        return "content word"          # digits carry content, whether or not word-initial
    if not t.startswith(" "):
        return "subword piece"
    return "function word" if s.lower() in STOP else "content word"


ORDER = ["function word", "content word", "punctuation & space", "subword piece"]
COL = {"function word": "#4C72B0", "content word": "#55A868",
       "punctuation & space": "#DD5B45", "subword piece": "#8A6BBE"}

# ---------------------------------------------------------------- top panel data
a = np.load(T + "slideviz_L200h.npz", allow_pickle=True)
b = np.load(T + "slideviz_sswa_L200h.npz", allow_pickle=True)
toks = [str(x) for x in a["tokens"]]
W, L = int(a["W"]), int(a["L"])
dp = np.asarray(b["trained_prob_win"], float) - np.asarray(a["trained_prob_win"], float)
X0, X1 = 104, min(L, 156)                       # one sentence span, fits the left panel

# ------------------------------------------------------------- bottom panel data
def _sweep(k):
    big = T + "probsweep_%s_big.npz" % k
    return np.load(big if os.path.exists(big) else T + "probsweep_%s.npz" % k)


sw = {k: _sweep(k) for k in ("swa0", "swa1", "sswa0", "sswa1")}
P = {k: sw[k]["probs"][:, W:] for k in sw}
TOKS = np.load(T + "probsweep_toks%s.npz" % ("_big" if os.path.exists(T + "probsweep_swa0_big.npz") else ""), allow_pickle=True)["toks"][:, W:]
EFF = (P["sswa0"] + P["sswa1"]) / 2 - (P["swa0"] + P["swa1"]) / 2
N1, N2 = P["swa1"] - P["swa0"], P["sswa1"] - P["sswa0"]
KL = np.array([[klass(str(t)) for t in row] for row in TOKS])

rows = []
for k in ORDER:
    m = KL == k
    n = int(m.sum())
    if n == 0:
        continue
    e = EFF[m]
    sem = e.std(ddof=1) / np.sqrt(n)
    noise = max(abs(N1[m].mean()), abs(N2[m].mean()))
    rows.append((k, n, e.mean(), sem, noise))

# ------------------------------------------------------------------------ figure
# Read the passage, do not decode a bar chart: every token is drawn as text on a patch
# shaded by its own dp, so where the two loss rules differ is legible in the sentence.
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

X0, X1 = 104, min(L, 168)
seg = [(toks[t], float(dp[t])) for t in range(X0, X1)]
PMAX = max(abs(v) for _, v in seg)
norm = TwoSlopeNorm(vmin=-PMAX, vcenter=0.0, vmax=PMAX)
cmap = plt.get_cmap("RdBu_r")

NCOL = 108                                   # characters per line at this font size
lines, cur = [], []
col = 0
for tk, v in seg:
    w = max(len(tk), 1)
    if col + w > NCOL:
        lines.append(cur); cur = []; col = 0
    cur.append((col, w, tk, v)); col += w
if cur:
    lines.append(cur)

fig = plt.figure(figsize=(figstyle.FULL, 1.05 + 0.22 * len(lines)))
gs = fig.add_gridspec(2, 2, height_ratios=[0.30 * len(lines), 1.0], width_ratios=[1.7, 1.0],
                      hspace=0.75, wspace=0.10)
ax = fig.add_subplot(gs[0, :])
for r, ln in enumerate(lines):
    for c, w, tk, v in ln:
        ax.add_patch(Rectangle((c, -r - 0.44), w, 0.88, fc=cmap(norm(v)), ec="none", zorder=1))
        ax.text(c + w / 2.0, -r, tk.replace(" ", "·"), family="monospace",
                fontsize=4.6, ha="center", va="center", color="0.05", zorder=2)
ax.set_xlim(-0.5, NCOL + 0.5); ax.set_ylim(-len(lines) + 0.4, 0.75)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("one WikiText passage, shaded by $\\Delta p$ (T-SWA $-$ SWA)",
             fontsize=figstyle.FS_TITLE - 1.0, pad=3, loc="left", color="0.25")

cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal",
                  fraction=0.16, pad=0.06, aspect=45)
cb.set_ticks([-PMAX, 0, PMAX])
cb.set_ticklabels(["$-$%.2f" % PMAX, "0", "$+$%.2f" % PMAX])
cb.ax.tick_params(labelsize=figstyle.FS_TICK - 1.0, length=2)
cb.outline.set_linewidth(0.5)

# class summary: the same difference averaged, against the same-rule seed spread
bx = fig.add_subplot(gs[1, 0])
x = np.arange(len(rows))
nz = max(r[4] for r in rows)
bx.axhline(0, color="0.45", lw=0.9, zorder=2)
for xx, (k, n, e, sem, noise) in zip(x, rows):
    bx.bar(xx, 2 * noise, width=0.86, bottom=-noise, color="0.90", zorder=0, lw=0)
    bx.bar(xx, e, width=0.52, color=COL[k], edgecolor="black", linewidth=0.5, zorder=3)
    bx.errorbar(xx, e, yerr=sem, fmt="none", ecolor="0.15", elinewidth=0.9,
                capsize=2.0, capthick=0.8, zorder=4)
bx.set_xticks(x)
bx.set_xticklabels(["function", "content", "punct.", "subword"], fontsize=figstyle.FS_TICK - 0.8)
bx.set_ylim(-nz * 1.25, nz * 1.25)
bx.set_yticks([-round(nz, 2), 0, round(nz, 2)])
figstyle.clean(bx)
figstyle.yname(bx, r"mean $\Delta p$", pad=0.055)
bx.set_title("averaged by token class", fontsize=figstyle.FS_TITLE - 1.0, pad=3,
             loc="left", color="0.25")
bx.text(0.99, 0.97, "grey $=$ same-rule seed spread", transform=bx.transAxes,
        ha="right", va="top", fontsize=figstyle.FS_TICK - 1.0, color="0.45")

# legend for the class colours, in the free cell beside the summary
lg = fig.add_subplot(gs[1, 1]); lg.axis("off")
lg.legend(handles=[Patch(facecolor=COL[k], edgecolor="black", lw=0.5, label=k) for k in ORDER],
          loc="center left", frameon=False, fontsize=figstyle.FS_LEGEND,
          handlelength=1.0, handleheight=0.85, labelspacing=0.35, borderpad=0.0)

plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
