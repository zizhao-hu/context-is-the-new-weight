"""windowed_attention.png -- 2x2 paper figure in methods.html scheme style. All panels share a 10-col grid
(5 prefix + 5 content) so every cell is the same size and aligned. Row 1: causal (grows; cold start empty) |
sliding+history (constant W, cold start filled by real prior text). Row 2: uniform (constant W, all real;
fixed-length predict-last, runs on a recurrence) | sliding+trainable (cold start filled by a trainable prompt)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
CONTENT, HISTC, PROMPT = (.18, .53, .31), (.60, .81, .67), (.93, .50, .11)
PREF = ["t-4", "t-3", "t-2", "t-1", "t0"]; CONT = ["t1", "t2", "t3", "t4", "t5"]; PRED = ["t2", "t3", "t4", "t5", "t6"]
P, nreal, nrows, Wl = 5, 5, 5, 6; NC = P + nreal

def panel(a, title, mode, pad_color, pref_labels):
    img = np.zeros((nrows, NC, 4))
    for row in range(nrows):
        i = row + 1
        for idx in range(i): img[row, P + idx] = (*CONTENT, 1)
        if mode != "causal":
            npad = Wl - i
            for k in range(npad):
                col = P - npad + k; img[row, col] = (*pad_color, 1)
    a.imshow(img, aspect="equal", interpolation="nearest", zorder=2)
    for x in range(NC + 1): a.plot([x - .5, x - .5], [-.5, nrows - .5], color="#777", lw=1.0, zorder=3)
    for y in range(nrows + 1): a.plot([-.5, NC - .5], [y - .5, y - .5], color="#777", lw=1.0, zorder=3)
    a.set_xticks(range(NC)); a.set_xticklabels(pref_labels + CONT, fontsize=8.5, fontweight="bold")
    a.set_yticks(range(nrows)); a.set_yticklabels(["→" + t for t in PRED], fontsize=9, fontweight="bold")
    a.set_xlim(-.5, NC - .5); a.set_ylim(nrows - .5, -.5); a.tick_params(length=0, pad=2.5)
    a.set_title(title, fontsize=12.5, fontweight="bold", pad=5)
    for s in a.spines.values(): s.set_linewidth(2.0); s.set_color("#222")

fig, ax = plt.subplots(2, 2, figsize=(7.4, 4.4))
panel(ax[0, 0], "causal (grows)", "causal", None, PREF)
panel(ax[0, 1], "sliding + history", "win", HISTC, PREF)
panel(ax[1, 0], "uniform (constant $W$)", "win", CONTENT, PREF)
panel(ax[1, 1], "sliding + trainable", "win", PROMPT, [f"s{i+1}" for i in range(5)])
for a in (ax[0, 0], ax[1, 0]): a.set_ylabel("predicting next  ↓", fontsize=9.5, fontweight="bold")
fig.legend(handles=[Patch(color=CONTENT, label="real token"), Patch(color=HISTC, label="history fill"),
                    Patch(color=PROMPT, label="trainable prompt")],
           loc="upper center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.06), frameon=False, handlelength=1.4, columnspacing=1.6)
plt.tight_layout(rect=[0, 0, 1, 0.92], h_pad=1.6, w_pad=1.2)
out = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/.claude/worktrees/exp/paper/attention-sink/figures/windowed_attention.png"
plt.savefig(out, dpi=170, bbox_inches="tight"); print("wrote", out)
