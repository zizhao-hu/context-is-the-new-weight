#!/usr/bin/env python3
"""Paper Fig 5: what the symmetric loss rule does to next-token probability.

Top: per-token change in p(token | window), S-SWA minus SWA, on one real WikiText
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
        return "whitespace"
    if re.fullmatch(r"[^\w\s]+", s):
        return "punctuation"
    if s.isdigit():
        return "digit"
    if not t.startswith(" "):
        return "subword piece"
    return "function word" if s.lower() in STOP else "content word"


ORDER = ["function word", "content word", "punctuation", "digit", "subword piece"]
COL = {"function word": "#4C72B0", "content word": "#55A868", "punctuation": "#DD5B45",
       "digit": "#C39B3E", "subword piece": "#8A6BBE", "whitespace": "0.7"}

# ---------------------------------------------------------------- top panel data
a = np.load(T + "slideviz_L200h.npz", allow_pickle=True)
b = np.load(T + "slideviz_sswa_L200h.npz", allow_pickle=True)
toks = [str(x) for x in a["tokens"]]
W, L = int(a["W"]), int(a["L"])
dp = np.asarray(b["trained_prob_win"], float) - np.asarray(a["trained_prob_win"], float)
X0, X1 = 104, min(L, 178)                       # the two-sentence span used earlier

# ------------------------------------------------------------- bottom panel data
sw = {k: np.load(T + "probsweep_%s.npz" % k) for k in ("swa0", "swa1", "sswa0", "sswa1")}
P = {k: sw[k]["probs"][:, W:] for k in sw}
TOKS = np.load(T + "probsweep_toks.npz", allow_pickle=True)["toks"][:, W:]
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
plt.rcParams.update({"font.size": 11})
fig, (ax, bx) = plt.subplots(2, 1, figsize=(13.2, 5.8),
                             gridspec_kw={"height_ratios": [1.5, 1.0]})

PMAX = float(np.abs(dp[X0:X1]).max())
ax.axhline(0, color="0.55", lw=0.8, zorder=2)
for t in range(X0, X1):
    k = klass(toks[t])
    ax.add_patch(Rectangle((t + 0.08, 0), 0.84, float(dp[t]), fc=COL[k], ec="none", zorder=3))
    ax.text(t + 0.5, -PMAX * 1.12, toks[t].replace(" ", "·"), fontsize=7.4, rotation=90,
            ha="center", va="top", color=COL[k])
ax.set_xlim(X0 - 0.6, X1 + 0.4)
ax.set_ylim(-PMAX * 1.95, PMAX * 1.15)
ax.set_yticks([-round(PMAX, 1), 0, round(PMAX, 1)])
ax.tick_params(labelsize=9.5, length=3)
ax.set_ylabel(r"$\Delta p$  (S-SWA $-$ SWA)", fontsize=10.5)
ax.set_xticks([])
for sp in ("top", "right", "bottom"):
    ax.spines[sp].set_visible(False)
ax.legend(handles=[Patch(facecolor=COL[k], label=k) for k in ORDER],
          loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=5, frameon=False,
          fontsize=10, handlelength=1.1, handleheight=0.9, columnspacing=1.4,
          handletextpad=0.4)

y = np.arange(len(rows))[::-1]
nz = max(r[4] for r in rows) * 1.05
bx.axvspan(-nz, nz, color="0.88", zorder=0)
bx.axvline(0, color="0.45", lw=0.9, zorder=2)
for yy, (k, n, e, sem, noise) in zip(y, rows):
    bx.barh(yy, e, height=0.62, color=COL[k], zorder=3)
    bx.errorbar(e, yy, xerr=sem, fmt="none", ecolor="0.15", elinewidth=1.0,
                capsize=2.5, capthick=1.0, zorder=4)
    bx.text(nz * 0.60, yy, "%+.3f  (n=%d)" % (e, n), fontsize=9.5, va="center", ha="left", color="0.25")
bx.set_yticks(y)
bx.set_yticklabels([r[0] for r in rows], fontsize=10.5)
bx.set_xlim(-nz * 1.15, nz * 1.15)
bx.set_xlabel(r"mean $\Delta p$ by token class, 40 held-out chunks, two seeds per rule "
              r"(grey band $=$ same-rule seed difference)", fontsize=10.5)
bx.tick_params(labelsize=9.5, length=3)
for sp in ("top", "right", "left"):
    bx.spines[sp].set_visible(False)

fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.13, hspace=0.42)
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("wrote", OUT)
for k, n, e, sem, noise in rows:
    print("  %-15s n=%6d  eff %+.4f +- %.4f   seed-noise %+.4f" % (k, n, e, sem, noise))
