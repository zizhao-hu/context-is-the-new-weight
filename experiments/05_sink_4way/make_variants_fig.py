"""Grouped bars: position-1 sink rate for each scheme, pretrained text (wikitext) vs instruction QA (alpaca)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
schemes = ["base", "full", "sliding_history", "startup"]
labels = ["base", "causal", "sliding with\nhistory", "sliding with\ntrainable tokens"]


def load(s, d):
    f = os.path.join(EXP, f"sink/{s}{d}.json")
    return json.load(open(f))["sink_rate_pos1"] * 100 if os.path.exists(f) else 0


wt = [load(s, "") for s in schemes]
al = [load(s, "_alpaca") for s in schemes]
x = np.arange(len(schemes)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.5))
b1 = ax.bar(x - w / 2, wt, w, label="pretrained text (wikitext)", color="#6c7a89")
b2 = ax.bar(x + w / 2, al, w, label="instruction QA (alpaca)", color="#c0392b")
for b in (b1, b2):
    for r in b:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, f"{r.get_height():.0f}%", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("position-1 sink rate (% heads with mean attn > 0.3)")
ax.set_title("Position-1 sink — pretrained text vs instruction QA (same conclusion)", fontsize=11)
ax.set_ylim(0, 72); ax.legend(); ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
out = EXP + "/sink_variants.png"
plt.savefig(out, dpi=145, bbox_inches="tight"); print("wrote", out)
