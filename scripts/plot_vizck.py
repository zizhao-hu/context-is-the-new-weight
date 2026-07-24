"""Where does the sliding model's freed pos-0 mass go? Full-causal vs sliding, from trained checkpoints.
Row1: mean-over-heads attention heatmap. Row2: one deep query's attention over keys (char-labelled)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import PowerNorm
T = "/Users/zizhaohu/.claude/jobs/dad80856/tmp"
da = np.load(f"{T}/vizck_a.npz", allow_pickle=True); db = np.load(f"{T}/vizck_b.npz", allow_pickle=True)
def disp(c): return {" ": "␣", "\n": "\\n", "\t": "\\t"}.get(c, c)
fig, ax = plt.subplots(2, 2, figsize=(11, 7), gridspec_kw=dict(height_ratios=[1.25, 1]))
for col, (d, name) in enumerate([(da, "Full-causal"), (db, "Sliding")]):
    M = d["Asink"]; toks = list(d["toks"]); W = int(d["W"]); Lm = int(d["Lm"])
    p0 = float(d["psink"]); loc = float(d["lsink"]); sh = int(d["sh"])
    axh = ax[0, col]
    im = axh.imshow(M, aspect="auto", cmap="magma", norm=PowerNorm(0.45, vmin=0, vmax=min(0.6, M.max())))
    axh.set_title(f"{name}  (sink head, L{sh//8}.H{sh%8})\npos-0 mass={p0:.3f}  local-8 mass={loc:.3f}", fontsize=11, fontweight="bold")
    axh.set_xlabel("key position", fontsize=10); axh.set_ylabel("query position", fontsize=10)
    if col == 0: axh.axvline(0.5, color="cyan", lw=1.0, ls="--", alpha=.8)
    fig.colorbar(im, ax=axh, fraction=0.046, pad=0.02)
    axb = ax[1, col]; q = Lm - 1; rowv = M[q]
    ks = np.arange(0, q + 1) if name == "Full-causal" else np.arange(max(0, q - W + 1), q + 1)
    axb.bar(np.arange(len(ks)), rowv[ks], width=1.0, color="#2b6cb0")
    if name == "Full-causal":
        axb.bar([0], [rowv[0]], width=1.0, color="#d1495b")
        axb.annotate("pos-0", (0, rowv[0]), textcoords="offset points", xytext=(3, 0), fontsize=9, color="#d1495b", fontweight="bold")
    axb.set_title(f"deep query (pos {q}) attention over keys", fontsize=10, fontweight="bold")
    axb.set_ylabel("attention (mean head)", fontsize=9.5)
    step = max(1, len(ks) // 30); idx = np.arange(0, len(ks), step)
    axb.set_xticks(idx); axb.set_xticklabels([disp(toks[ks[i]]) for i in idx], fontsize=6.5)
    axb.set_xlim(-0.6, len(ks) - 0.4); axb.spines[["top", "right"]].set_visible(False)
fig.suptitle("Where the pos-0 mass goes: full-causal parks it on token 0; sliding has no far anchor $\\to$ disperses over the window (local $+$ common tokens)", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/figures/sink_investigation/10_where_mass_goes.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
