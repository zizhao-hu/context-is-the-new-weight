"""Triangle-vs-sliding sink distribution, two regimes side by side, COMPACT.
Equal bar width in both panels; left (from-scratch, 2 masks) narrower than right (CPT, 3 models)
via width_ratios matched to x-span. Separate y-axes: toy sink is ~25x weaker in absolute terms
(4-layer char GPT, max-head) than CPT (real LLM, mean-head) -> a shared axis flattens the toy to 0."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.gridspec import GridSpec

# toy (BOS, Lm=1024, max head): mask -> (p0, p0sem, boundary, bdsem)
TOY = {"Full-causal": (0.0237, 0.0022, 0.0006, 0.0000),
       "Sliding":     (0.0000, 0.0000, 0.0075, 0.0003)}
# CPT (full deploy, mean head, n=12): model -> (p0, p0sem, dist, distsem)
CPT = {"Base":        (0.5855, 0.0031, 0.0479, 0.0011),
       "Full-causal": (0.6404, 0.0046, 0.0537, 0.0018),
       "Sliding":     (0.1630, 0.0267, 0.5017, 0.0311)}
COL = {"Base": "#9aa0a6", "Full-causal": "#2b6cb0", "Sliding": "#d1495b"}

W = 1.0          # bar width (data units) -- IDENTICAL in both panels
GAP = 1.3        # gap between the two groups (pos-0 / relocated)

def layout(models):
    n = len(models); span = 2 * (n * W) + GAP        # 2 groups
    g0 = np.arange(n) * W                              # group-1 bar x's
    g1 = g0 + n * W + GAP                              # group-2 bar x's
    return g0, g1, span, n

gL0, gL1, spanL, nL = layout(list(TOY))
gC0, gC1, spanC, nC = layout(list(CPT))

fig = plt.figure(figsize=(6.6, 2.7))
gs = GridSpec(1, 2, width_ratios=[spanL, spanC], wspace=0.28)
axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1])

plt.rcParams.update({"font.weight": "bold", "axes.linewidth": 1.4})
BW = {"fontweight": "bold"}
def draw(ax, data, g0, g1, span, n, glabels, ylab, title):
    for i, m in enumerate(data):
        p0, p0s, d, ds = data[m]
        ax.bar([g0[i], g1[i]], [p0, d], W, yerr=[p0s, ds], color=COL[m],
               capsize=3, error_kw=dict(lw=1.6), label=m)
    ax.set_xticks([g0.mean(), g1.mean()]); ax.set_xticklabels(glabels, fontsize=11, **BW)
    ax.set_xlim(g0[0] - 0.7, g1[-1] + 0.7)
    ax.set_ylabel(ylab, fontsize=11, **BW); ax.set_title(title, fontsize=11.5, **BW)
    ax.spines[["top", "right"]].set_visible(False)
    for t in ax.get_yticklabels(): t.set_fontweight("bold")
    ax.tick_params(labelsize=10, width=1.4)

draw(axL, TOY, gL0, gL1, spanL, nL, ["pos-0", "boundary"],
     "sink mass (max head)", "From scratch (toy GPT)")
draw(axR, CPT, gC0, gC1, spanC, nC, ["pos-0", "common-token"],
     "attn mass (mean head)", "Continued pretraining (Llama-3B)")
# full 3-entry legend, top-right of FIRST panel (relocated bars are tiny -> space is free)
h, l = axR.get_legend_handles_labels()
axL.legend(h, l, fontsize=9.5, frameon=False, loc="upper right",
           labelspacing=0.35, handlelength=1.2, prop={"weight": "bold"})
fig.tight_layout()
out = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/figures/sink_investigation/8_triangle_vs_sliding_2panel.png"
fig.savefig(out, dpi=160, bbox_inches="tight"); print("saved", out)
