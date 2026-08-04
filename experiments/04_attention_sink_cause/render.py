"""Render the attention-sink comparison: per-layer sink + key-position profile, base vs experience-tuned."""
import json, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
res = {}
for f in sorted(glob.glob(os.path.join(EXP, "sink_*.json"))):
    d = json.load(open(f)); res[d["label"]] = d
order = [k for k in ["base", "experience", "windowed"] if k in res]
col = {"base": "#888", "experience": "#c0392b", "windowed": "#2e7d32"}

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))

# per-layer sink
a = ax[0]; nL = len(res[order[0]]["per_layer_sink"]); x = np.arange(nL); w = 0.8 / len(order)
for i, k in enumerate(order):
    a.bar(x + i*w, res[k]["per_layer_sink"], w, label=f"{k} (mean {res[k]['sink_overall']:.3f})", color=col[k])
a.set_xticks(x + w*(len(order)-1)/2); a.set_xticklabels([f"L{j+1}" for j in range(nL)])
a.set_xlabel("softmax layer (of 8)"); a.set_ylabel("attention mass on key pos 0 (sink)")
a.set_title("Per-layer attention sink"); a.legend(fontsize=8.5)

# key-position profile
b = ax[1]
for k in order:
    p = res[k]["key_position_profile"]; b.plot(range(len(p)), p, "o-", color=col[k], label=k, ms=3.5)
b.set_xlabel("key position"); b.set_ylabel("mean attention mass")
b.set_title("Attention by key position — sink is concentrated at position 0")
b.legend(fontsize=8.5); b.grid(alpha=0.25)

plt.tight_layout()
out = os.path.join(EXP, "attention_sink.png")
plt.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)
