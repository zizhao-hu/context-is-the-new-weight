"""Three attention-pattern diagrams for instruction tuning on one query→answer example, in proper next-token
form: each ROW is the token being PREDICTED, attending to everything BEFORE it (predicting A2 uses Q1→A1).
Green-boxed rows (→A1 →A2 →A3) are the trained answer predictions; the number on the right = working-memory
size (how many tokens that prediction attends).
  1 full-causal — context GROWS 1→5: later predictions attend more (non-uniform working memory).
  2 windowed    — constant-width parallelogram, full query in; the cold-start left is filled by the LAST FEW
                  REAL EXAMPLES (their own query=light-blue / answer=light-green history).
  3 startup     — the SAME parallelogram, but the cold-start left is filled by TRAINABLE PROMPTS (no prior
                  examples needed). Both give uniform working memory with the full query kept.
(Exp-05 trained on plain wikitext; Q/A is the illustrative form.)"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

QUERY, ANS, PROMPT = (.27, .55, .80), (.30, .70, .35), (.90, .49, .13)
HISTQ, HISTA = (.58, .74, .92), (.64, .86, .64)      # previous-example query / answer (lighter = history)
COLS_REAL = ["Q1", "Q2", "Q3", "A1", "A2"]           # attended tokens t0..t4 (A3 predicted but never attended)
PRED = ["Q2", "Q3", "A1", "A2", "A3"]                # rows: predict t1..t5, each from the tokens before it
nQ = 3                                               # t0..t2 are the current query
# previous-example tail filling the 5 padding slots: …pQ2 pQ3 pA1 pA2 pA3 (2 prev-query + 3 prev-answer)
PREV_COLORS = [HISTQ, HISTQ, HISTA, HISTA, HISTA]
PREV_LABELS = ["q", "q", "a", "a", "a"]


def panel(a, title, mode, P, Wl, pad_colors, pad_labels):
    nreal, nrows = len(COLS_REAL), len(PRED)
    NC = P + nreal
    img = np.zeros((nrows, NC, 4))
    counts = []
    for row in range(nrows):
        i = row + 1                                  # predicting token t_i; prior context = t0..t_{i-1} (query always kept)
        real_idx = list(range(0, i))
        npad = (Wl - i) if (mode == "pad" and i < Wl) else 0
        counts.append(npad + len(real_idx))
        for k in range(npad):                        # last npad padding slots fill the cold start up to width Wl
            col = P - npad + k
            img[row, col] = (*pad_colors[col], .9)
        for idx in real_idx:
            img[row, P + idx] = (*(QUERY if idx < nQ else ANS), .9)
    a.imshow(img, aspect="equal", interpolation="nearest", zorder=2)
    for x in range(NC + 1):
        a.plot([x - .5, x - .5], [-.5, nrows - .5], color="#ccc", lw=.5, zorder=3)
    for y in range(nrows + 1):
        a.plot([-.5, NC - .5], [y - .5, y - .5], color="#ccc", lw=.5, zorder=3)
    a.add_patch(Rectangle((-.5, 2 - .5), NC, 3, fill=False, ec="#1b5e20", lw=1.1, zorder=4))   # →A1 →A2 →A3
    for r, c in enumerate(counts):
        a.text(NC - .3, r, str(c), va="center", ha="left", fontsize=5.5,
               color="#1b5e20" if r >= 2 else "#888", fontweight="bold" if r >= 2 else "normal")
    xlab = list(pad_labels) + COLS_REAL
    a.set_xticks(range(NC)); a.set_xticklabels(xlab, fontsize=4.6)
    a.set_yticks(range(nrows)); a.set_yticklabels(["→" + t for t in PRED], fontsize=5)
    for t, lab in zip(a.get_yticklabels(), PRED):
        t.set_color("#1b5e20" if lab.startswith("A") else "#888")
    a.set_xlim(-.5, NC + .4); a.set_ylim(nrows - .5, -.5)
    a.tick_params(length=2, pad=1.5)
    a.set_title(title, fontsize=7); a.set_xlabel("attends to →", fontsize=5.5)


fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.9), gridspec_kw={"width_ratios": [5, 10, 10]})
panel(ax[0], "causal", "full", 0, 99, [], [])
panel(ax[1], "sliding with history", "pad", 5, 6, PREV_COLORS, PREV_LABELS)
panel(ax[2], "sliding with trainable tokens", "pad", 5, 6, [PROMPT] * 5, [f"s{i+1}" for i in range(5)])
ax[0].set_ylabel("predicting next token ↓", fontsize=5.5)
fig.legend(handles=[Patch(color=HISTQ, label="prev-example query"), Patch(color=HISTA, label="prev-example answer"),
                    Patch(color=PROMPT, label="trainable startup prompt"), Patch(color=QUERY, label="current query"),
                    Patch(color=ANS, label="current answer")], loc="upper center", ncol=5, fontsize=5.3,
           bbox_to_anchor=(0.5, 1.04), frameon=False, handlelength=1.1, columnspacing=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.9])
out = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/.claude/worktrees/exp/experiments/05_sink_4way/qa_schemes.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
