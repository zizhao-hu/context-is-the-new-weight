#!/usr/bin/env python3
"""fig:toystream, FAIR version: all masks streamed to 30k at the matched W=128 budget.
Literature A-D dashed; ours (E S-SWA, F +prefix regs, G +riding regs) solid.
Parses CURVE lines from curves128.txt (harvested from the fair-run logs)."""
import re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "/Users/zizhaohu/.claude/jobs/f25a34dc/tmp/curves128.txt"
txt = open(SRC).read()
def get(kind, tag):
    m = re.search(r"CURVE%s %s (means|sems) = \[([^\]]+)\]" % ("_REGS" if kind == "regs" else "", tag), txt)
    return None
means, sems = {}, {}
for pre, tag, key in re.findall(r"CURVE(_REGS)? (\w+) (means|sems) = \[([^\]]+)\]", txt.replace("] ", "]\n"))[:0]:
    pass
for mm in re.finditer(r"CURVE(_REGS)? (\w+) (means|sems) = \[([^\]]+)\]", txt):
    kind = "regs" if mm.group(1) else "plain"
    k = (kind, mm.group(2))
    vals = [float(x) for x in mm.group(4).split(",")]
    (means if mm.group(3) == "means" else sems)[k] = vals

ORDER = [(("plain", "a_w128"),   "A. full causal",          "--", "#7f7f7f"),
         (("plain", "b_w128"),   "B. SWA",                  "--", "#1f77b4"),
         (("plain", "c_w128"),   "C. SWAA",                 "--", "#2ca02c"),
         (("plain", "e_B_w128"), "D. Transformer-XL",       "--", "#d62728"),
         (("plain", "b_w128_fw"),"E. S-SWA",                "-",  "#9467bd"),
         (("regs", "k_w128_fw"), "F. $+$sink token",        "-",  "#e377c2"),
         (("regs", "m_w128_fw"), "F. $+$sink prefix",       "-",  "#ff7f0e"),
         (("regs", "n_w128_fw"), "G. $+$riding sink token", "-",  "#8c564b"),
         (("regs", "o_w128_fw"), "G. $+$riding sink prefix","-",  "#17becf")]
x = np.arange(1, 31)
plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
fig, ax = plt.subplots(figsize=(6.8, 3.4))
for key, label, ls, col in ORDER:
    if key not in means:
        print("MISSING", key); continue
    m = np.array(means[key]); s = np.array(sems.get(key, [0] * len(m)))
    ax.plot(x, m, ls, color=col, lw=1.9, label="%s (%.1f)" % (label, m[-1]))
    ax.fill_between(x, m - s, m + s, color=col, alpha=0.13, lw=0)
ax.set_ylim(33, 50); ax.set_xlim(0.5, 30.5)
ax.set_xlabel("stream position (k tokens)", fontsize=12)
ax.set_ylabel("perplexity", fontsize=12)
ax.legend(fontsize=8.0, frameon=False, ncol=2, loc="upper right",
          title="training mask (ppl at 30k)", title_fontsize=8.6)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=10)
fig.tight_layout()
out = "paper/attention-sink/figures/toy_stream_ppl.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
