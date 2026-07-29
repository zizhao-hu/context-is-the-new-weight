"""One figure style for the whole paper.

The rule that matters: author each figure at the size it is rendered at. A single-column
figure is included at \\linewidth (3.34in) and a full-width one at \\textwidth (7.0in), so a
figure authored larger than that gets scaled down and its text shrinks with it. That is what
made the same nominal font size come out at 5.5pt in one figure and 8pt in another.

Everything else here is the bold, high-contrast look: heavy bars with white separations,
value chips, no top or right spine, and the axis name inside the axes so the left margin is
not wasted.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL, FULL = 3.34, 7.0                      # \linewidth and \textwidth in this template

FS_TICK, FS_AXIS, FS_TITLE, FS_LEGEND, FS_CHIP = 6.5, 7.5, 8.0, 7.0, 7.0
LW_AXES, LW_BAR = 0.9, 0.6

# shared palette: p0 sink, trainable register, distributed sink, content
C_P0, C_REG, C_SEP, C_CT = "#4C72B0", "#DD5B45", "#55A868", "#D3D3D3"


def apply():
    plt.rcParams.update({
        "font.size": FS_TICK,
        "axes.linewidth": LW_AXES,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,
        "hatch.linewidth": 0.35,
        "hatch.color": "0.15",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def clean(ax):
    """no top or right spine, short ticks"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.0, labelsize=FS_TICK)


def yname(ax, text, pad=0.075, also=()):
    """Axis name running vertically just inside the left spine.

    The left margin then carries only the tick numbers. `pad` (a fraction of the axes width)
    is added to the left of the data range first, so the name sits in an empty strip instead
    of on top of the leftmost bar. Call this after the data and any explicit xlim are set.
    """
    for a in (ax,) + tuple(also):          # pad siblings too so panel widths stay identical
        x0, x1 = a.get_xlim()
        if a.get_xscale() == "log":
            a.set_xlim(x0 / (x1 / x0) ** (pad / (1 - pad)), x1)
        else:
            a.set_xlim(x0 - (x1 - x0) * pad / (1 - pad), x1)
    return ax.text(pad * 0.42, 0.5, text, transform=ax.transAxes, rotation=90,
                   ha="center", va="center", fontsize=FS_AXIS)


def yticks_inside(ax, x=0.085, size=None):
    """Move the y tick numbers inside the axes, so the left margin disappears entirely.

    Call after yname(), which reserves the strip they sit in.
    """
    ax.figure.canvas.draw()                       # tick label text is only filled in on draw
    ys, labs = ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()]
    lo, hi = ax.get_ylim()
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    tr = ax.get_yaxis_transform()                 # x in axes fraction, y in data units
    for yv, lb in zip(ys, labs):
        if lo <= yv <= hi and lb:
            ax.text(x, yv, lb, transform=tr, ha="left", va="center", fontsize=size or FS_TICK)


def chip(ax, x, y, v, fmt="%.2f"):
    """bold value label on a white rounded plate, readable on top of any fill"""
    return ax.text(x, y, fmt % v, ha="center", va="center", fontsize=FS_CHIP,
                   color="0.1", fontweight="bold", zorder=7,
                   bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="0.45",
                             lw=0.7, alpha=0.92))
