"""Bar chart: position-1 attention sink (mean attention to first content token + sink-rate) across the base
model and the 4 long-context tuning schemes."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.abspath(__file__))
order = ["base", "full", "sliding_history", "startup"]    # the 3 figure configs (causal / sliding-history / sliding-trainable) + base
labels = {"base": "base\n(reference)", "full": "causal", "sliding_history": "sliding with\nhistory",
          "startup": "sliding with\ntrainable tokens"}
colors = {"base": "#888", "full": "#c0392b", "sliding_history": "#16a085", "startup": "#2e6da4"}
res = {k: json.load(open(os.path.join(EXP, f"sink/{k}.json"))) for k in order if os.path.exists(os.path.join(EXP, f"sink/{k}.json"))}
ks = [k for k in order if k in res]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
a = ax[0]; vals = [res[k]["mean_attn_first_content"] for k in ks]
a.bar(range(len(ks)), vals, color=[colors[k] for k in ks])
for i, v in enumerate(vals):
    a.text(i, v + max(vals) * 0.02, f"{v:.3f}", ha="center", fontsize=8)
a.set_xticks(range(len(ks))); a.set_xticklabels([labels[k] for k in ks], fontsize=8)
a.set_ylabel("mean attention to first content token")
a.set_title("Position-1 sink — mean attention\n(full-attention eval; lower = less sink)", fontsize=10)
a.grid(alpha=0.25, axis="y")

b = ax[1]; vals = [res[k]["sink_rate_pos1"] * 100 for k in ks]
b.bar(range(len(ks)), vals, color=[colors[k] for k in ks])
for i, v in enumerate(vals):
    b.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
b.set_xticks(range(len(ks))); b.set_xticklabels([labels[k] for k in ks], fontsize=8)
b.set_ylabel("% heads with mean attn to token 1 > 0.3")
b.set_title("Position-1 sink rate (paper's metric)", fontsize=10)
b.grid(alpha=0.25, axis="y")

plt.tight_layout()
out = os.path.join(EXP, "sink4_comparison.png")
plt.savefig(out, dpi=145, bbox_inches="tight"); print("wrote", out)
