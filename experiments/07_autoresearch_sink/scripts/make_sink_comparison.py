import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

contexts = [128, 256, 512, 1024, 2048]
x = np.arange(len(contexts))
# sink rate = % of heads whose mean attention to token 1 exceeds eps=0.3, reproduced from
# Wang et al. 2025 (arXiv:2504.02732) Fig 5a (trend; 128 ~ 0, "nearly non-existent").
sink = [1, 15, 25, 40, 50]
# data-side: under plain causal attention the first token's *warranted* share of attention is 2/C (shrinks with C).
data = [2.0 / C * 100 for C in contexts]            # %  (1.56 -> 0.10)

fig, ax = plt.subplots(figsize=(5.4, 4.3))
# bars: the model's attention-sink rate (grows)
ax.bar(x, sink, 0.56, color="#7a4fa3", edgecolor="#3f2860", lw=0.8, zorder=3, label="attention sink (model)")
for xi, v in enumerate(sink):
    ax.text(xi, v + 1.4, f"{v}%", ha="center", fontsize=11, fontweight="bold", color="#3f2860")
ax.set_ylim(0, 60); ax.set_xticks(x); ax.set_xticklabels(contexts, fontsize=12)
ax.set_xlabel("training context length  $C$", fontsize=13)
ax.set_ylabel("attention-sink rate  (% of heads)", color="#3f2860", fontsize=13, fontweight="bold")
ax.tick_params(axis="y", labelcolor="#3f2860", labelsize=11)
ax.grid(alpha=0.25, axis="y", zorder=0)
# line on a log right axis: first-token importance warranted by the data (shrinks)
ax2 = ax.twinx()
ax2.plot(x, data, "o--", color="#c0392b", lw=3.0, ms=11, zorder=4, label="first-token weight (data $\\propto 1/C$)")
ax2.set_yscale("log"); ax2.set_ylim(0.07, 2.4)
ax2.set_ylabel("first-token share warranted\nby the data  (%, log)", color="#c0392b", fontsize=13, fontweight="bold")
ax2.tick_params(axis="y", labelcolor="#c0392b", labelsize=11)
# divergence annotations
ax.annotate("model sinks MORE", xy=(3.9, 50), xytext=(1.05, 55),
            fontsize=11.5, fontweight="bold", color="#3f2860",
            arrowprops=dict(arrowstyle="->", color="#7a4fa3", lw=2))
ax2.annotate("data warrants LESS", xy=(2.55, 0.40), xytext=(0.30, 0.95), ha="left",
             fontsize=11.5, fontweight="bold", color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
ax.set_title("The model sinks \\emph{against} the data\n(sink grows as 1/$C$ importance shrinks): a mask artifact".replace("\\emph{","").replace("}",""),
             fontsize=12.5, fontweight="bold")
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
fig.text(0.5, -0.02, "sink rate reproduced from Wang et al.\\ 2025 (arXiv:2504.02732)".replace("\\ "," "),
         ha="center", fontsize=8, color="#777")
plt.tight_layout()
out = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/.claude/worktrees/exp/figures/scaling/sink_comparison.png"
plt.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.12); print("wrote", out)
