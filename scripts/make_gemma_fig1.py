"""Paper Fig 1 (Gemma-3-4B two sink types) in the unified Fig-6 style:
magma + log color, 32-token bin means, bold titles, query/key axis names
(no numeric ticks), top+bottom panel edges only, no colorbar.
Data: gemma_slidev.npz (global_attn = full-attention layers mean,
local_attn = SWA layers mean, W=sw)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

NPZ = "/Users/zizhaohu/.claude/jobs/f25a34dc/tmp/gemma_slidev.npz"
OUT = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/figures/gemma_slidev.png"
d = np.load(NPZ, allow_pickle=True)
G, Lc, W = d["global_attn"].astype(np.float32), int(d["L"]), int(d["sw"])
Loc = d["local_attn"].astype(np.float32)
T = G.shape[0]; BIN = 32; NB = T // BIN

q = np.arange(T)[:, None]; k = np.arange(T)[None, :]
causal = k <= q
slide = causal & (k > q - W)

def binmap(m, valid):
    v = m.copy(); v[~valid] = np.nan
    r = v.reshape(NB, BIN, NB, BIN).swapaxes(1, 2).reshape(NB, NB, -1)
    with np.errstate(invalid="ignore"):
        return np.nanmean(r, axis=2)

norm = LogNorm(vmin=2e-4, vmax=0.5, clip=True)
PANELS = [("Gemma-3-4B full-attention layers", binmap(G, causal)),
          ("Gemma-3-4B sliding-window layers", binmap(Loc, slide))]

fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
for ax, (title, v) in zip(axes, PANELS):
    ax.imshow(v, extent=[0, T, T, 0], cmap="magma", norm=norm,
              interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=17, fontweight="bold", pad=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("key", fontsize=16, fontweight="bold")
    for side, on in (("top", True), ("bottom", True), ("left", False), ("right", False)):
        ax.spines[side].set_visible(on); ax.spines[side].set_linewidth(2.2)
axes[0].set_ylabel("query", fontsize=16, fontweight="bold")

# callouts (fig-1 identity): p0 sink column in full layers; separator columns in SWA layers
BLUE, ORNG = "#2b6cb0", "#e07b18"
kw = dict(fontsize=14.5, fontweight="bold", va="center",
          bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none", lw=0))
axes[0].annotate("p0 sink\n(Peng et al. 2026)", xy=(30, 900), xytext=(430, 330),
                 color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.6), **kw)
band = binmap(Loc, slide)
colmean = np.nanmean(np.nan_to_num(band), 0)
colmean[:2] = 0                                              # skip the leading edge
sep_bins = np.argsort(colmean)[-2:]
for sb in sep_bins:
    sc = sb * BIN + BIN / 2
    qy = min(sc + 500, T - 60)                               # a query row where this key is in-window
    axes[1].annotate("", xy=(sc, qy), xytext=(1080, 430),
                     arrowprops=dict(arrowstyle="->", color=ORNG, lw=2.6))
axes[1].text(1080, 380, "distributed sink\n(Ruscio et al. 2025)", color=ORNG, ha="center", **kw)
fig.subplots_adjust(left=0.045, right=0.985, top=0.9, bottom=0.075, wspace=0.08)
fig.savefig(OUT, dpi=200)
print("saved", OUT)
