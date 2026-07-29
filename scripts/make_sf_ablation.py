#!/usr/bin/env python3
"""SWA -> T-SWA loss-mask ablation as BAR charts with trend curves.

Left: sink distribution per dose -- stacked bars (p0 + separator) with trend curves
      tracing the p0 drain and the total.
Right: perplexity per dose -- bars for short-context ppl (the rising cost) with its trend
       curve, and a flat line+band for deep streaming ppl (the axis that never moves).
Data: figures/sf_ablation.tsv (measured; refuses to plot without it).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DATA = "figures/sf_ablation.tsv"
MIX = "figures/sf_mix.tsv"
OUT_SINK = "paper/attention-sink/figures/sf_sink.png"
OUT_PPL  = "paper/attention-sink/figures/sf_ppl.png"
if not (os.path.exists(DATA) and os.path.exists(MIX)):
    sys.exit("missing %s or %s" % (DATA, MIX))

rows = [[float(x) for x in l.split("\t")] for l in open(DATA) if l.strip() and not l.startswith("#")]
rows.sort()
sf, ppl, ppl_sem, p0, p0s, sep, seps, short = map(np.array, zip(*rows))
mrows = [[float(x) for x in l.split("\t")] for l in open(MIX) if l.strip() and not l.startswith("#")]
mrows.sort()
mal, mppl, mppl_sem, mp0, msep, mshort = map(np.array, zip(*mrows))
GAP = 1.6
mx = len(sf) - 1 + GAP + 1 + np.arange(len(mal))          # mixing block positions
# append mixing block to the shared arrays (sink sems <=0.003, caps omitted)
x = np.concatenate([np.arange(len(sf)), mx])
p0 = np.concatenate([p0, mp0]); sep = np.concatenate([sep, msep])
p0s = np.concatenate([p0s, np.zeros(len(mal))]); seps = np.concatenate([seps, np.zeros(len(mal))])
ppl = np.concatenate([ppl, mppl]); ppl_sem = np.concatenate([ppl_sem, mppl_sem])
short = np.concatenate([short, mshort])
tot = p0 + sep
nsf = len(sf)
labels = ["%d" % v for v in sf] + ["0" if a == 0 else ("1" if a == 1 else ("%.2f" % a).lstrip("0").rstrip("0")) for a in mal]

C_P0, C_SEP, C_SHORT, C_STREAM = "#4C72B0", "#55A868", "#B04C4C", "#555555"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
f1, a1 = plt.subplots(figsize=(3.34, 2.5))       # single column
f2, a2 = plt.subplots(figsize=(3.34, 2.5))

# ---- left: sink distribution ----
w = 0.72
a1.bar(x, p0, w, color=C_P0, edgecolor="white", linewidth=0.6)
a1.bar(x, sep, w, bottom=p0, color=C_SEP, edgecolor="white", linewidth=0.6)
for sl in (slice(0, nsf), slice(nsf, None)):
    a1.plot(x[sl], p0[sl], "-o", color="#1f3d63", ms=4, lw=1.8, zorder=5)
    a1.plot(x[sl], tot[sl], "-o", color="#2e6b45", ms=4, lw=1.8, zorder=5)
a1.errorbar(x, p0, yerr=p0s, fmt="none", ecolor="0.15", elinewidth=0.9, capsize=2, zorder=6)
a1.errorbar(x, tot, yerr=np.sqrt(p0s**2 + seps**2), fmt="none", ecolor="0.15",
            elinewidth=0.9, capsize=2, zorder=6)
a1.set_ylabel("attention mass", fontsize=8)
a1.set_ylim(0, max(tot) * 1.28)

a1.legend(handles=[Patch(facecolor=C_P0, label="p0 sink"),
                   Patch(facecolor=C_SEP, label="distributed sink")],
          frameon=False, fontsize=6.6, ncol=2, loc="upper right", borderpad=0.1,
          handlelength=1.2, columnspacing=0.9, handletextpad=0.4)

# ---- right: perplexity (grouped bars: in-context vs streaming) ----
bw = 0.38
a2.bar(x - bw / 2, short, bw, color=C_SHORT, edgecolor="white", linewidth=0.5, alpha=0.9)
a2.bar(x + bw / 2, ppl, bw, color=C_STREAM, edgecolor="white", linewidth=0.5, alpha=0.9)
for sl in (slice(0, nsf), slice(nsf, None)):
    a2.plot(x[sl] - bw / 2, short[sl], "-o", color="#7a2f2f", ms=4, lw=1.8, zorder=5)
    a2.plot(x[sl] + bw / 2, ppl[sl], "-o", color="#2b2b2b", ms=4, lw=1.8, zorder=5)
a2.errorbar(x + bw / 2, ppl, yerr=ppl_sem, fmt="none", ecolor="0.1",
            elinewidth=0.9, capsize=2, zorder=6)
a2.set_ylabel("perplexity", fontsize=8)
a2.set_ylim(30, max(short) * 1.15)

a2.legend(handles=[Patch(facecolor=C_SHORT, label="in-context (len 64)"),
                   Patch(facecolor=C_STREAM, label="streaming (30k)")],
          frameon=False, fontsize=6.6, ncol=2, loc="upper center", borderpad=0.1,
          handlelength=1.1, columnspacing=0.9, handletextpad=0.4)

div = nsf - 1 + GAP / 2 + 0.5
for ax in (a1, a2):
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=5.6, rotation=90)
    ax.axvline(div, color="0.75", lw=0.9, ls=":")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, labelsize=7)
    ax.text((nsf - 1) / 2, -0.34, "unscored rows", ha="center", fontsize=7,
            transform=ax.get_xaxis_transform())
    ax.text(np.mean(mx), -0.34, "mix $\\alpha$", ha="center", fontsize=7,
            transform=ax.get_xaxis_transform())

for f, o in ((f1, OUT_SINK), (f2, OUT_PPL)):
    f.tight_layout()
    f.savefig(o, dpi=300, bbox_inches="tight")
    print("wrote", o)
