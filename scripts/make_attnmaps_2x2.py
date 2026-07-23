"""Paper Fig 6 (attn maps, 2x2) from REAL attention values (attn4.npz, dumped by
attn4_dump.py on the cluster: Qwen2.5-0.5B, W=256, passage chosen for strongest
base p0 sink). Panels: base full / base sliding / windowed CPT / startup CPT.
32-token bin means of row-normalised attention; log color scale; borderless.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

NPZ = "/Users/zizhaohu/.claude/jobs/f25a34dc/tmp/attn4.npz"
OUT = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/figures/attn_binned_2x2.png"
T, W, P, BIN = 1536, 256, 256, 32
NBIN = T // BIN
d = np.load(NPZ)

def binmap(m, valid, blank_rows=0):
    v = m.astype(np.float32).copy()
    v[~valid] = np.nan
    if blank_rows: v[:blank_rows] = np.nan
    r = v.reshape(NBIN, BIN, NBIN, BIN).swapaxes(1, 2).reshape(NBIN, NBIN, -1)
    with np.errstate(invalid="ignore"):
        return np.nanmean(r, axis=2)

q = np.arange(T)[:, None]; k = np.arange(T)[None, :]
causal = k <= q
slide = causal & (k > q - W)
slide_pfx = (causal & (k > q - W)) | (causal & (k < P))     # startup: prompt always visible

PANELS = [
    ("full context",                    binmap(d["base_full"], causal)),
    ("sliding window",                  binmap(d["base_slide"], slide)),
    ("symmetric sliding window",        binmap(d["windowed"], slide, blank_rows=W)),
    ("$+$ trainable sink prefix",       binmap(d["startup"], slide_pfx, blank_rows=P)),
]
ANNOT = {2: "history accum.\n(no prediction)", 3: "trainable prompt\n(no prediction)"}
CYAN = "#1899c2"
norm = LogNorm(vmin=2e-4, vmax=0.5, clip=True)

fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.2))
for j, (ax, (title, v)) in enumerate(zip(axes.flat, PANELS)):
    ax.imshow(v, extent=[0, T, T, 0], cmap="magma", norm=norm,
              interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=17.5, fontweight="bold", pad=7)
    ax.set_xticks([0, 500, 1000, 1500]); ax.set_yticks([0, 500, 1000, 1500])
    ax.tick_params(labelsize=15, length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    if j in ANNOT:
        ax.add_patch(Rectangle((8, 8), P - 16, P - 16, fill=False, edgecolor=CYAN, lw=2.5))
        ax.text(P + 44, 30, ANNOT[j], color=CYAN, fontsize=15.5, fontweight="bold", va="top")
for ax in axes[0, :]: ax.set_xticklabels([])
for ax in axes[:, 1]: ax.set_yticklabels([])
for ax in axes[1, :]: ax.set_xticklabels(["0", "500", "1000", ""])
fig.subplots_adjust(left=0.075, right=0.885, top=0.945, bottom=0.045, wspace=0.055, hspace=0.15)
cax = fig.add_axes([0.9, 0.15, 0.028, 0.7])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="magma"), cax=cax)
cb.set_ticks([1e-1, 1e-2, 1e-3])
cb.set_ticklabels(["$10^{-1}$", "$10^{-2}$", "$10^{-3}$"])
cb.outline.set_visible(False)
cax.tick_params(labelsize=16, length=0)
fig.savefig(OUT, dpi=200)
print("saved", OUT)
