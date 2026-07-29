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
plt.rcParams.update({"font.size": 10.5, "axes.linewidth": 0.9})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.1),
                             gridspec_kw={"width_ratios": [2.55, 1.0], "wspace": 0.30})

PMAX = float(np.abs(dp[X0:X1]).max())
ax.axhline(0, color="0.55", lw=0.8, zorder=2)
for t in range(X0, X1):
    k = klass(toks[t])
    ax.add_patch(Rectangle((t + 0.08, 0), 0.84, float(dp[t]), fc=COL[k], ec="none", zorder=3))
    ax.text(t + 0.5, -PMAX * 1.10, toks[t].replace(" ", "\u00b7"), fontsize=7.2, rotation=90,
            ha="center", va="top", color=COL[k])
ax.set_xlim(X0 - 0.6, X1 + 0.4)
ax.set_ylim(-PMAX * 2.05, PMAX * 1.12)
ax.set_yticks([-round(PMAX, 1), 0, round(PMAX, 1)])
ax.tick_params(labelsize=9, length=3)
ax.set_ylabel(r"$\Delta p$   (T-SWA $-$ SWA)", fontsize=10)
ax.set_xticks([])
for sp in ("top", "right", "bottom"):
    ax.spines[sp].set_visible(False)
ax.set_title("per token, one passage", fontsize=10.5, pad=6, loc="left", color="0.25")

x = np.arange(len(rows))
nz = max(r[4] for r in rows)
bx.axhline(0, color="0.45", lw=0.9, zorder=2)
for xx, (k, n, e, sem, noise) in zip(x, rows):
    bx.bar(xx, 2 * noise, width=0.84, bottom=-noise, color="0.90", zorder=0, lw=0)
    bx.bar(xx, e, width=0.58, color=COL[k], zorder=3)
    bx.errorbar(xx, e, yerr=sem, fmt="none", ecolor="0.15", elinewidth=1.0,
                capsize=2.2, capthick=0.9, zorder=4)
bx.set_xticks(x)
SHORT = {"function word": "function\nword", "content word": "content\nword",
         "punctuation & space": "punct.\n& space", "subword piece": "subword\npiece"}
bx.set_xticklabels([SHORT[r[0]] for r in rows], fontsize=8.8, linespacing=1.2)
for xx, r in zip(x, rows):
    bx.text(xx, -nz * 0.93, "n=%s" % (("%.1fk" % (r[1] / 1000)) if r[1] >= 1000 else r[1]),
            fontsize=7.6, ha="center", va="center", color="0.45")
bx.set_ylim(-nz * 1.12, nz * 1.12)
bx.set_yticks([-round(nz, 2), 0, round(nz, 2)])
bx.tick_params(labelsize=9, length=3)
bx.set_ylabel(r"mean $\Delta p$", fontsize=10)
bx.set_title("held-out chunks, 2 runs per rule", fontsize=10.5, pad=6, loc="left", color="0.25")
bx.text(len(rows) - 0.52, nz * 0.97, "grey $=$ same-rule\nseed spread, per class", fontsize=8.2,
        ha="right", va="top", color="0.45", linespacing=1.15)
for sp in ("top", "right", "bottom"):
    bx.spines[sp].set_visible(False)

fig.legend(handles=[Patch(facecolor=COL[k], label=k) for k in ORDER],
           loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False,
           fontsize=10, handlelength=1.1, handleheight=0.9, columnspacing=1.6,
           handletextpad=0.4)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
for k, n, e, sem, noise in rows:
    print("  %-15s n=%6d  eff %+.4f +- %.4f   seed-noise %+.4f" % (k, n, e, sem, noise))
