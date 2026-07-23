"""Rebuild paper Fig 6 (attn maps) as a 2x2 from the original 6-panel attn_binned.png.

The original generator is lost. Panels are recovered as VALUES, not pixels: each
16-token bin is a flat-colored cell, so we sample cell centers, invert the colormap
(sampled from the original colorbar, log-spaced) back to attention values, aggregate
2x2 cells into 32-token bins by averaging, and re-render with the same cmap/norm.
Kept panels: base+full, base+sliding, sliding w/ history CPT, sliding w/ prefix CPT.
Cyan setup-region annotations are redrawn natively. Borderless style.
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

SRC = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/figures/attn_binned.png"
OUT = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/figures/attn_binned_2x2.png"

im = np.array(Image.open(SRC).convert("RGB"))
spread = im.astype(int).max(2) - im.astype(int).min(2)
colorful = spread > 25
colcount = colorful.sum(0)
blocks, inb = [], False
for x in range(im.shape[1]):
    if colcount[x] > 5 and not inb: start, inb = x, True
    elif colcount[x] <= 5 and inb: blocks.append((start, x)); inb = False
if inb: blocks.append((start, im.shape[1]))
assert len(blocks) == 7, blocks
Y0, Y1 = 94, 544

# colormap + log geometry from the original colorbar (1e-1 at y=182, decade = 162 px)
bx0, bx1 = blocks[6]
bar = im[57:550, (bx0 + bx1) // 2, :].astype(float) / 255.0     # top -> bottom
cmap = ListedColormap(bar[::-1])
e_top = -1 + (182 - 57) / 162.0
e_bot = -1 - (550 - 182) / 162.0
norm = LogNorm(vmin=10 ** e_bot, vmax=10 ** e_top)
bar_vals = 10 ** np.linspace(e_top, e_bot, len(bar))            # value of each bar row (top->bottom)

NB = 96                                                          # original 16-token bins
def panel_values(i):
    x0, x1 = blocks[i]
    # interior: skip the black spine + antialiased edge if the block includes it
    ix0 = x0
    while im[100:540, ix0].astype(int).mean() < 100: ix0 += 1
    ix0 += 1                                                     # antialiased gray column
    px = im[97:544, ix0:x1].astype(float)
    H, W = px.shape[:2]
    cy = ((np.arange(NB) + 0.5) * H / NB).astype(int)
    cx = ((np.arange(NB) + 0.5) * W / NB).astype(int)
    cells = px[np.ix_(cy, cx)]                                   # [NB, NB, 3] center colors
    flat = cells.reshape(-1, 3)
    white = flat.min(1) > 235
    gray = ((flat.max(1) - flat.min(1)) < 20) & (flat.max(1) > 80)   # bright gray: spines/antialiasing (dark low-spread = data)
    cyan = (flat[:, 2] > 120) & (flat[:, 2] - flat[:, 0] > 40) & (flat[:, 1] - flat[:, 0] > 20)
    d = ((flat[:, None, :] / 255.0 - bar[None, :, :]) ** 2).sum(-1)
    vals = bar_vals[d.argmin(1)]
    vals[white | gray | cyan] = np.nan
    vals = vals.reshape(NB, NB)
    q, k = np.meshgrid(np.arange(NB), np.arange(NB), indexing="ij")
    vals[k > q] = np.nan                                         # future is masked by construction
    return vals

def coarsen(v, f=2):                                             # 16-token -> 32-token bins
    n = NB // f
    with np.errstate(invalid="ignore"):
        c = np.nanmean(v.reshape(n, f, n, f).swapaxes(1, 2).reshape(n, n, f * f), axis=2)
    return c

TITLES = {0: "full context", 1: "sliding window",
          3: "sliding window $+$ ours", 4: "ours $+$ trainable sink prefix"}
ANNOT = {3: "history accum.\n(no prediction)", 4: "trainable prompt\n(no prediction)"}
ORDER = [0, 1, 3, 4]
CYAN = "#1899c2"

fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.2))
for ax, i in zip(axes.flat, ORDER):
    v = coarsen(panel_values(i))
    ax.imshow(v, extent=[0, 1536, 1536, 0], cmap=cmap, norm=norm,
              interpolation="nearest", aspect="equal")
    ax.set_title(TITLES[i], fontsize=17.5, fontweight="bold", pad=7)
    ax.set_xticks([0, 500, 1000, 1500]); ax.set_yticks([0, 500, 1000, 1500])
    ax.tick_params(labelsize=15, length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    if i in ANNOT:
        ax.add_patch(Rectangle((8, 8), 248, 248, fill=False, edgecolor=CYAN, lw=2.5))
        ax.text(300, 30, ANNOT[i], color=CYAN, fontsize=15.5, fontweight="bold", va="top")
for ax in axes[0, :]: ax.set_xticklabels([])
for ax in axes[:, 1]: ax.set_yticklabels([])
for ax in axes[1, :]: ax.set_xticklabels(["0", "500", "1000", ""])
fig.subplots_adjust(left=0.075, right=0.885, top=0.945, bottom=0.045, wspace=0.055, hspace=0.15)
cax = fig.add_axes([0.9, 0.15, 0.028, 0.7])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_ticks([1e-1, 1e-2, 1e-3])
cb.outline.set_visible(False)
cax.tick_params(labelsize=16, length=0)
fig.savefig(OUT, dpi=200)
print("saved", OUT)
