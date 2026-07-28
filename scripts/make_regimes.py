"""Regenerate paper Figure 2 (regimes.png): attention-mask panels in ONE compact row.
Literature (a. Triangle, b. SWA, c. SWAA, d. Transformer-XL) then IN THIS WORK (e,f,g).

T-SWA (e): ONE chunk of the standard training size C, window W = C/2, ordinary sliding causal
band, NO carried history. Loss is taken on every query that has a FULL window, i.e. q >= W-1 =
C/2-1 -- context = C/2-1 unscored rows, loss = C/2+1 scored rows, every scored token sees exactly
W = C/2 real tokens (removes the (W-1)/L context starvation of naive SWA).
f: T-SWA + a FIXED trainable sink (token/prefix) -- always-attended prepended register (vertical).
g: T-SWA + a RIDING trainable sink (token/prefix) -- attended at a CONSTANT relative offset, so it
   rides the window's trailing (oldest) edge (diagonal) rather than sitting at a fixed position.

Run: python scripts/make_regimes.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

blue = (0.18, 0.43, 0.71); lblue = (0.72, 0.81, 0.90)      # loss rows / context-only rows
orange = (0.90, 0.55, 0.15); lorange = (0.96, 0.83, 0.62)
dk = "#333"
W = 3; h = 2; nP = 1; Q = 6          # Q = C (standard chunk, 6x6), W = C/2
LOSS0 = W - 1                        # first query with a full W-window -> loss on q >= C/2-1
TFS = 13; HFS = 16; BFS = 11
OUT = ["/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/figures/regimes.png",
       "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/.claude/worktrees/exp/paper/attention-sink/figures/regimes.png"]

fig, ax = plt.subplots(figsize=(17, 3.6))
ry = 0.0

def panel(ox, title, ncols, fn, brackets=(), loss_from=None):
    for q in range(Q):
        y = ry + (Q - 1 - q)
        for c in range(ncols):
            col = fn(q, c)
            ax.add_patch(Rectangle((ox + c, y), 1, 1, facecolor=(col or "white"), edgecolor="black", lw=0.8))
    ax.add_patch(Rectangle((ox, ry), ncols, Q, fill=False, edgecolor="black", lw=2.4, zorder=5))
    if ". " in title:                              # bold the index only, name stays regular
        _idx, _name = title.split(". ", 1)
        title = r"$\mathbf{%s.}$ %s" % (_idx, _name)
    ax.text(ox + ncols / 2.0, ry + Q + 0.25, title, ha="center", va="bottom", fontsize=TFS, fontweight="normal")
    for br in brackets:
        a, b, txt, col = br[:4]
        ax.plot([ox + a, ox + b], [ry - 0.45, ry - 0.45], color=col, lw=2)
        ax.text(ox + (a + b) / 2.0, ry - 0.75, txt, ha="center", va="top", fontsize=BFS, color=col, fontweight="bold")
    if loss_from is not None:                      # left bracket marking the scored rows
        ytop = ry + (Q - 1 - loss_from) + 1
        ax.plot([ox - 0.4, ox - 0.4], [ry, ytop], color=dk, lw=2.2)
        ax.text(ox - 0.85, (ytop + ry) / 2.0, "loss", ha="center", va="center",
                fontsize=BFS, color=dk, rotation=90, fontweight="bold")

def inwin(absp, q): return (q - W < absp <= q)
def band(q, c, off=0):
    if q < LOSS0: return None        # context rows: not scored -> leave their row white
    cc = c - off
    if not inwin(cc, q): return None
    return blue
def band_ride(q, c, off=0):          # T-SWA window (blue) + an ADDITIONAL riding register (orange)
    if q < LOSS0: return None        # attended one step OLDER than the window, at a CONSTANT offset,
    cc = c - off                     # so it rides the trailing edge as a separate slot (its own column)
    if inwin(cc, q): return blue
    if cc == q - W:  return orange   # register: just older than the oldest in-window token
    return None

# ---- single row: literature (a,b,c,d) then in-this-work (e,f,g) ---------------------
G, GBIG = 1.3, 3.0
x = 0.0; POS = {}
def place(key, ncols, gap=G):
    global x
    x += gap if POS else 0
    POS[key] = (x, ncols); x += ncols
    return POS[key][0]

ox = place("a", Q, 0);  panel(ox, "A. Triangle", Q, lambda q, c: blue if c <= q else None)
ox = place("b", Q);     panel(ox, "B. SWA", Q, lambda q, c: blue if inwin(c, q) else None)
ox = place("c", Q);     panel(ox, "C. SWAA", Q, lambda q, c: blue if (c == 0 or inwin(c, q)) else None,
                              brackets=[(0, 1, "sink", dk)])
ox = place("d", Q + 2); panel(ox, "D. Transformer-XL", Q + 2, lambda q, c: blue if c <= q + 2 else None,
                              brackets=[(0, 2, "history", dk)])
ox = place("e", Q, GBIG); panel(ox, "E. T-SWA", Q, lambda q, c: band(q, c), loss_from=LOSS0)
ox = place("f", nP + Q, G);  panel(ox, "E $+$ fixed sink", nP + Q,
                                lambda q, c: (orange if q >= LOSS0 else None) if c < nP else band(q, c, nP),
                                brackets=[(0, nP, "trainable sink", orange)], loss_from=LOSS0)
ox = place("g", nP + Q, G);  panel(ox, "E $+$ riding sink", nP + Q,
                              lambda q, c: band_ride(q, c, nP),
                              brackets=[(0, nP, "trainable sink", orange)], loss_from=LOSS0)
T = x

# group labels
lit_x0 = POS["a"][0]; lit_x1 = POS["d"][0] + POS["d"][1]
our_x0 = POS["e"][0] - 1.6; our_x1 = POS["g"][0] + POS["g"][1]
ax.text((lit_x0 + lit_x1) / 2.0, ry + Q + 2.05, "Existing literature", ha="center", fontsize=HFS, fontweight="bold")
ax.text((our_x0 + our_x1) / 2.0, ry + Q + 2.05, "This work: truncated sliding window attention",
        ha="center", fontsize=HFS, fontweight="bold")
# divider between literature and ours
xd = (POS["d"][0] + POS["d"][1] + POS["e"][0] - 1.6) / 2.0
ax.plot([xd, xd], [ry - 1.2, ry + Q + 1.7], color="#bbb", lw=1.6, ls="--", zorder=1)

# future-region zigzag
def fzig(ox, off, q0=0):
    pts = [(ox + off + q0 + 1, ry + Q - q0)]
    for q in range(q0, Q):
        xx = ox + off + q + 1
        pts.append((xx, ry + Q - 1 - q))
        if q < Q - 1: pts.append((xx + 1, ry + Q - 1 - q))
    a, b = zip(*pts); ax.plot(a, b, color="black", lw=2.0, zorder=6)
# on e/f/g the zigzag only tracks the scored rows -- the context rows are blank
for k, off, q0 in [("a", 0, 0), ("b", 0, 0), ("c", 0, 0), ("d", 2, 0),
                   ("e", 0, LOSS0), ("f", nP, LOSS0), ("g", nP, LOSS0)]:
    fzig(POS[k][0], off, q0)
# d alone has a history / current-chunk split worth boxing
ax.add_patch(Rectangle((POS["d"][0] + (Q + 2) - Q, ry), Q, Q, fill=False, edgecolor="black", lw=2.8, zorder=7))

ax.set_xlim(-1.3, T + 0.2); ax.set_ylim(-1.6, ry + Q + 2.6); ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout()
for o in OUT: plt.savefig(o, dpi=200, bbox_inches="tight", pad_inches=0.02, facecolor="white")

# hard-crop to the true ink bbox with equal small margins (bbox_inches leaves an uneven left margin)
from PIL import Image
import numpy as np
for o in OUT:
    im = Image.open(o).convert("RGB"); arr = np.asarray(im.convert("L"))
    ink = arr < 250
    rows = np.where(ink.any(axis=1))[0]; cols = np.where(ink.any(axis=0))[0]
    pad = 6
    x0 = max(cols[0] - pad, 0); x1 = min(cols[-1] + pad + 1, im.width)
    y0 = max(rows[0] - pad, 0); y1 = min(rows[-1] + pad + 1, im.height)
    im.crop((x0, y0, x1, y1)).save(o)
print("wrote regimes.png (a,b,c,d literature | e,f,g in-this-work T-SWA; W=C/2, loss on q>=C/2-1)")
