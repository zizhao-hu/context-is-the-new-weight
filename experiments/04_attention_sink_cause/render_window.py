"""Render: does the sliding window reduce the sink? frontier sink (attention to the oldest-visible key) for
full causal vs W=64 vs W=128, base and experience-tuned, against the uniform-attention baseline."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
res = {lab: json.load(open(os.path.join(EXP, f"sinkwin_{lab}.json"))) for lab in ["base", "experience"]}
wins = ["0", "64", "128"]; wlab = ["full causal", "window 64", "window 128"]
col = {"base": "#888", "experience": "#c0392b"}

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))

# frontier sink (the apples-to-apples sink: oldest-visible key)
a = ax[0]; x = np.arange(len(wins)); w = 0.38
for i, lab in enumerate(["base", "experience"]):
    vals = [res[lab]["by_window"][k]["frontier_sink"] for k in wins]
    bars = a.bar(x + i*w, vals, w, label=lab, color=col[lab])
    for b, v in zip(bars, vals):
        a.text(b.get_x()+b.get_width()/2, v+0.001, f"{v:.3f}", ha="center", fontsize=8)
# uniform-attention references for the windowed cases
a.axhline(1/64, ls=":", color="#1b7a1b", lw=1); a.text(1.5, 1/64+0.0006, "uniform (1/64)", color="#1b7a1b", fontsize=7.5)
a.axhline(1/128, ls=":", color="#2e6da4", lw=1); a.text(2.0, 1/128-0.0024, "uniform (1/128)", color="#2e6da4", fontsize=7.5)
a.set_xticks(x + w/2); a.set_xticklabels(wlab)
a.set_ylabel("frontier sink — attention to oldest-visible key")
a.set_title("Sliding window REDUCES the sink\n(full ≈ 0.055 ≫ windowed ≈ 0.003–0.008 ≤ uniform: no new edge sink)")
a.legend(fontsize=9)

# absolute pos-0 sink (mechanically masked away under a window)
b = ax[1]
for i, lab in enumerate(["base", "experience"]):
    vals = [res[lab]["by_window"][k]["sink_abs0"] for k in wins]
    b.bar(x + i*w, vals, w, label=lab, color=col[lab])
b.set_xticks(x + w/2); b.set_xticklabels(wlab)
b.set_ylabel("attention to absolute key position 0")
b.set_title("Absolute pos-0 sink\n(window masks key 0 for far queries → collapses)")
b.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(EXP, "sliding_window_sink.png")
plt.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)
