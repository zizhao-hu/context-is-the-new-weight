"""windowed_attention.png for the paper -- 2x2 in the methods.html scheme style. Each row = the token being
predicted, attending to everything in its window. Row 1: causal (context grows) | sliding with history (constant
W, cold start filled by PRIOR real text). Row 2: uniform window (constant W, every token exactly W REAL tokens via
fixed-length predict-last; runs on a recurrence) | sliding with trainable tokens (cold start filled by a trainable
prompt that slides out)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
CONTENT, HISTC, PROMPT = (.30, .62, .42), (.66, .84, .70), (.90, .49, .13)
COLS = ["t1", "t2", "t3", "t4", "t5"]
PRED = ["t2", "t3", "t4", "t5", "t6"]

def panel(a, title, mode, P, Wl, pad_colors, pad_labels):
    nreal, nrows = len(COLS), len(PRED); NC = P + nreal
    img = np.zeros((nrows, NC, 4)); counts = []
    for row in range(nrows):
        i = row + 1; real_idx = list(range(0, i)); npad = (Wl - i) if (mode == "pad" and i < Wl) else 0
        counts.append(npad + len(real_idx))
        for k in range(npad):
            col = P - npad + k; img[row, col] = (*pad_colors[col], .9)
        for idx in real_idx:
            img[row, P + idx] = (*CONTENT, .9)
    a.imshow(img, aspect="equal", interpolation="nearest", zorder=2)
    for x in range(NC + 1): a.plot([x - .5, x - .5], [-.5, nrows - .5], color="#ccc", lw=.5, zorder=3)
    for y in range(nrows + 1): a.plot([-.5, NC - .5], [y - .5, y - .5], color="#ccc", lw=.5, zorder=3)
    for r, c in enumerate(counts):
        a.text(NC - .3, r, str(c), va="center", ha="left", fontsize=6, color="#1b5e20", fontweight="bold")
    a.set_xticks(range(NC)); a.set_xticklabels(list(pad_labels) + COLS, fontsize=5.2)
    a.set_yticks(range(nrows)); a.set_yticklabels(["→" + t for t in PRED], fontsize=5.6, color="#1b5e20")
    a.set_xlim(-.5, NC + .4); a.set_ylim(nrows - .5, -.5); a.tick_params(length=2, pad=1.5)
    a.set_title(title, fontsize=8); a.set_xlabel("attends to →", fontsize=6)

fig, ax = plt.subplots(2, 2, figsize=(6.6, 4.2))
panel(ax[0,0], "causal (context grows)", "full", 0, 99, [], [])
panel(ax[0,1], "sliding with history (real prior text)", "pad", 5, 6, [HISTC]*5, ["t-4","t-3","t-2","t-1","t0"])
panel(ax[1,0], "uniform window (constant $W$, all real)", "pad", 5, 6, [CONTENT]*5, ["t-4","t-3","t-2","t-1","t0"])
panel(ax[1,1], "sliding with trainable tokens", "pad", 5, 6, [PROMPT]*5, [f"s{i+1}" for i in range(5)])
ax[0,0].set_ylabel("predicting next token ↓", fontsize=6)
ax[1,0].set_ylabel("predicting next token ↓", fontsize=6)
fig.legend(handles=[Patch(color=CONTENT, label="real token (predicted / context)"),
                    Patch(color=HISTC, label="prior text (history fill)"),
                    Patch(color=PROMPT, label="trainable prompt")],
           loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.04), frameon=False, handlelength=1.2, columnspacing=1.2)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/.claude/worktrees/exp/paper/attention-sink/figures/windowed_attention.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
