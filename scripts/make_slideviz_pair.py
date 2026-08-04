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


sw = {k: _sweep(k) for k in ("swa0", "swa1", "sswa0", "sswa1", "a0", "a1", "tf0", "tf1")}
P = {k: sw[k]["probs"][:, W:] for k in sw}
TOKS = np.load(T + "probsweep_toks%s.npz" % ("_big" if os.path.exists(T + "probsweep_swa0_big.npz") else ""), allow_pickle=True)["toks"][:, W:]
EFF = (P["sswa0"] + P["sswa1"]) / 2 - (P["swa0"] + P["swa1"]) / 2
EFF2 = (P["tf0"] + P["tf1"]) / 2 - (P["a0"] + P["a1"]) / 2
N1, N2 = P["swa1"] - P["swa0"], P["sswa1"] - P["sswa0"]
N3, N4 = P["a1"] - P["a0"], P["tf1"] - P["tf0"]
KL = np.array([[klass(str(t)) for t in row] for row in TOKS])

rows = []
for k in ORDER:
    m = KL == k
    n = int(m.sum())
    if n == 0:
        continue
    e = EFF[m]
    e2 = EFF2[m]
    sem = e.std(ddof=1) / np.sqrt(n)
    sem2 = e2.std(ddof=1) / np.sqrt(n)
    noise = max(abs(N1[m].mean()), abs(N2[m].mean()),
                abs(N3[m].mean()), abs(N4[m].mean()))
    rows.append((k, n, e.mean(), sem, noise, e2.mean(), sem2))

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

NCOL = 88                                    # characters per line in the narrower left panel
lines, cur = [], []
col = 0
for tk, v in seg:
    w = max(len(tk), 1)
    if col + w > NCOL:
        lines.append(cur); cur = []; col = 0
    cur.append((col, w, tk, v)); col += w
if cur:
    lines.append(cur)

fig = plt.figure(figsize=(figstyle.FULL, 1.48))
gs = fig.add_gridspec(1, 2, width_ratios=[2.45, 1.0], wspace=0.14)
ax = fig.add_subplot(gs[0, 0])
for r, ln in enumerate(lines):
    for c, w, tk, v in ln:
        ax.add_patch(Rectangle((c, -r - 0.44), w, 0.88, fc=cmap(norm(v)), ec="none", zorder=1))
        ax.text(c + w / 2.0, -r, tk.replace(" ", "·"), family="monospace",
                fontsize=4.4, ha="center", va="center", color="0.05", zorder=2)
ax.set_xlim(-0.5, NCOL + 0.5); ax.set_ylim(-len(lines) + 0.35, 0.75)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("WikiText passage token probability",
             fontsize=figstyle.FS_TITLE - 1.0, pad=3, loc="left", color="0.25")

# colour key inline, in the space left over after the last token, so it costs no extra row
_r = len(lines) - 1
_end = lines[-1][-1][0] + lines[-1][-1][1]
_w = 15
_x0 = NCOL - _w - 7                          # right-aligned on the last line
if _x0 > _end + 4:
    _g = np.linspace(-PMAX, PMAX, 128)[None, :]
    ax.imshow(_g, extent=[_x0, _x0 + _w, -_r - 0.20, -_r + 0.20], cmap=cmap, norm=norm,
              aspect="auto", zorder=1)
    ax.add_patch(Rectangle((_x0, -_r - 0.20), _w, 0.40, fc="none", ec="0.35", lw=0.5, zorder=3))
    ax.text(_x0 - 1.4, -_r, "$-$%.2f" % PMAX, ha="right", va="center",
            fontsize=figstyle.FS_TICK - 1.8, color="0.35")
    ax.text(_x0 + _w + 1.4, -_r, "$+$%.2f" % PMAX, ha="left", va="center",
            fontsize=figstyle.FS_TICK - 1.8, color="0.35")

# class summary beside it; the tick labels carry the class colours, so no legend is needed
bx = fig.add_subplot(gs[0, 1])
x = np.arange(len(rows))
nz = max(max(r[4], abs(r[5])) for r in rows)
bx.axhline(0, xmin=0.115, color="0.45", lw=0.9, zorder=2)
for xx, (k, n, e, sem, noise, e2, sem2) in zip(x, rows):
    bx.bar(xx, 2 * noise, width=0.86, bottom=-noise, color="0.90", zorder=0, lw=0)
    bx.bar(xx - 0.185, e, width=0.34, color=COL[k], edgecolor="black", linewidth=0.5,
           zorder=3)
    bx.bar(xx + 0.185, e2, width=0.34, color=COL[k], edgecolor="black", linewidth=0.5,
           hatch="///", zorder=3)
    bx.errorbar(xx - 0.185, e, yerr=sem, fmt="none", ecolor="0.15", elinewidth=0.9,
                capsize=1.8, capthick=0.8, zorder=4)
    bx.errorbar(xx + 0.185, e2, yerr=sem2, fmt="none", ecolor="0.15", elinewidth=0.9,
                capsize=1.8, capthick=0.8, zorder=4)
bx.set_xticks([])                                # the legend names the classes
bx.set_ylim(-nz * 1.25, nz * 1.25)
bx.set_yticks([-round(nz, 2), 0, round(nz, 2)])
figstyle.clean(bx)
bx.tick_params(axis="y", labelsize=figstyle.FS_TICK - 1.0)
figstyle.yname(bx, r"mean $\Delta p$", pad=0.115, x=0.045)
bx.set_title("by token class", fontsize=figstyle.FS_TITLE - 1.0, pad=3, loc="left", color="0.25")
bx.text(0.99, 0.97, "grey $=$ noise floor; solid T-SWA$-$SWA, hatched TF$-$full",
        transform=bx.transAxes, ha="right", va="top",
        fontsize=figstyle.FS_TICK - 1.6, color="0.45")
FULL = {"function word": "function words", "content word": "content words",
        "punctuation & space": "punctuation", "subword piece": "subword continuation"}
bx.legend(handles=[Patch(facecolor=COL[k], edgecolor="black", lw=0.5, label=FULL[k])
                   for k, *_ in rows],
          loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
          fontsize=figstyle.FS_TICK - 1.2, handlelength=0.9, handleheight=0.8,
          columnspacing=0.8, handletextpad=0.35, labelspacing=0.25, borderpad=0.0)

_pa, _pb = ax.get_position(), bx.get_position()
_y0 = _pa.y0 + 0.105                          # room for the right panel's tick labels
bx.set_position([_pb.x0, _y0, _pb.width, (_pb.y0 + _pb.height) - _y0])
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
