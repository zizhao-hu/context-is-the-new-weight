#!/usr/bin/env python3
"""SWA -> S-SWA loss-mask ablation (Finding 3 dose-response), two panels.

x = number of unscored context rows (score_from): 0 = plain SWA ... W-1 = S-SWA.
Left: deep streaming perplexity (+/-1 SEM band) -- flat: the loss rule does not change LM quality.
Right: within-max-context sink masses (p0 / separator / total, +/-1 SEM) -- p0 drains monotonically.

All models: identical mask, W=128, C=256, data, and loss-token budget; ONLY score_from differs.
Data from figures/sf_ablation.tsv (measured; this script refuses to plot without it).
Columns: sf  ppl  ppl_sem  p0  p0_sem  sep  sep_sem   (within-max-context masses)
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

DATA = "figures/sf_ablation.tsv"
OUT = "paper/attention-sink/figures/sf_ablation.png"

if not os.path.exists(DATA):
    sys.exit("missing %s -- fill it from the sym_sf logs + sinkdecomp output first" % DATA)

rows = []
for ln in open(DATA, encoding="utf-8"):
    ln = ln.strip()
    if not ln or ln.startswith("#"):
        continue
    rows.append([float(x) for x in ln.split("\t")])
rows.sort()
sf, ppl, ppl_sem, p0, p0_sem, sep, sep_sem = map(np.array, zip(*rows))
tot = p0 + sep
tot_sem = np.sqrt(p0_sem**2 + sep_sem**2)

C_PPL, C_P0, C_SEP, C_TOT = "#555555", "#4C72B0", "#55A868", "#B04C4C"
plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0))

a1.fill_between(sf, ppl - ppl_sem, ppl + ppl_sem, color=C_PPL, alpha=0.18, lw=0)
a1.plot(sf, ppl, "-o", color=C_PPL, ms=4.5, lw=1.6)
a1.set_ylabel("streaming perplexity", fontsize=12)
a1.set_ylim(min(ppl) - 4, max(ppl) + 4)
a1.set_title("LM quality: flat", fontsize=12, fontweight="bold", pad=4)

for y, e, c, lab in ((p0, p0_sem, C_P0, "p0 sink"), (sep, sep_sem, C_SEP, "separator sink"),
                     (tot, tot_sem, C_TOT, "total sink")):
    a2.errorbar(sf, y, yerr=e, fmt="-o", color=c, ms=4.5, lw=1.6,
                elinewidth=1.0, capsize=2.4, label=lab)
a2.set_ylabel("attention mass", fontsize=12)
a2.set_ylim(0, max(tot) * 1.25)
a2.set_title("p0 drains; separator absorbs", fontsize=12, fontweight="bold", pad=4)
a2.legend(frameon=False, fontsize=10.5, handlelength=1.4, loc="center left")

W = int(max(sf)) + 1
for ax in (a1, a2):
    ax.set_xlabel("unscored context rows (score_from)", fontsize=12)
    ax.set_xticks([0, W // 4, W // 2, 3 * W // 4, W - 1])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.5, labelsize=10.5)
    lo, hi = ax.get_ylim()
    ax.text(0, lo + 0.02 * (hi - lo), "SWA", ha="left", va="bottom",
            fontsize=10.5, fontweight="bold", color="#333")
    ax.text(W - 1, lo + 0.02 * (hi - lo), "S-SWA", ha="right", va="bottom",
            fontsize=10.5, fontweight="bold", color="#333")

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
