"""Rebuild paper Fig 6 (attn maps) as a 2x2 from the original 6-panel attn_binned.png.

The original generator is lost; panels are extracted as bitmaps from the high-res PNG
(axes interiors detected by colorful-pixel bounding boxes) and recomposed with fresh
axes, titles, and a colorbar sampled from the original bar (so colors match exactly).
Kept panels: base+full, base+sliding, sliding w/ history CPT, sliding w/ prefix CPT.
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
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
PANELS = {i: im[Y0:Y1, blocks[i][0]:blocks[i][1]] for i in (0, 1, 3, 4)}

# colorbar: sample the original bar's colors bottom->top; log geometry from tick spacing
bx0, bx1 = blocks[6]
bar = im[57:550, (bx0 + bx1) // 2, :] / 255.0            # top -> bottom
cmap = ListedColormap(bar[::-1])                          # bottom -> top
# 1e-1 at y=182, one decade = 162 px  ->  bar top (y=57) and bottom (y=550) exponents
e_top = -1 + (182 - 57) / 162.0
e_bot = -1 - (550 - 182) / 162.0
norm = LogNorm(vmin=10 ** e_bot, vmax=10 ** e_top)

TITLES = {0: "full context", 1: "sliding window",
          3: "sliding window $+$ ours", 4: "ours $+$ trainable sink prefix"}
ORDER = [0, 1, 3, 4]

fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.6))
plt.rcParams.update({"font.size": 13})
for ax, i in zip(axes.flat, ORDER):
    ax.imshow(PANELS[i], extent=[0, 1536, 1536, 0], interpolation="bilinear", aspect="equal")
    ax.set_title(TITLES[i], fontsize=15, fontweight="bold", pad=8)
    ax.set_xticks([0, 500, 1000, 1500]); ax.set_yticks([0, 500, 1000, 1500])
    ax.tick_params(labelsize=11)
for ax in axes[:, 0]: ax.set_ylabel("query position", fontsize=13)
for ax in axes[1, :]: ax.set_xlabel("key position", fontsize=13)
for ax in axes[0, :]: ax.set_xticklabels([])
for ax in axes[:, 1]: ax.set_yticklabels([])
fig.subplots_adjust(left=0.09, right=0.88, top=0.95, bottom=0.08, wspace=0.06, hspace=0.14)
cax = fig.add_axes([0.905, 0.15, 0.025, 0.7])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_ticks([1e-1, 1e-2, 1e-3])
cax.tick_params(labelsize=12)
fig.savefig(OUT, dpi=200)
print("saved", OUT)
