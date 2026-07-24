"""CCD hill-climbing figure: prequential reward vs lifetime episode (smoothed), CCD modes vs frozen
baseline + static baseline levels. Minimal text, config bottom-right."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/eval_out"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

def rolling(xs, w=15):
    out = []
    for i in range(len(xs)):
        s = max(0, i - w + 1); out.append(sum(xs[s:i+1]) / (i - s + 1))
    return out

MODES = [("everything", "#2e7d32", "CCD: consolidate + time-clock (9+7)"),
         ("everything_notime", "#d62728", "consolidate, NO time-clock (9 only)"),
         ("hybrid", "#1f77b4", "CCD-hybrid (+recall)"),
         ("frozen", "#999999", "frozen — no consolidation")]
BASE = [("Act", 0.69, "#d62728"), ("full-hist ET", 0.61, "#9467bd")]

fig, ax = plt.subplots(figsize=(8.2, 5.0))
for m, c, lab in MODES:
    f = f"{R}/ccd_{m}.json"
    if not os.path.exists(f): continue
    rs = [r["reward"] for r in json.load(open(f)).get("log", [])]
    if not rs: continue
    ls = "--" if m == "everything_notime" else "-"
    ax.plot(range(1, len(rs)+1), rolling(rs), color=c, lw=2.2, ls=ls, label=lab)
for name, v, c in BASE:
    ax.axhline(v, color=c, ls=":", lw=1.5, alpha=0.8)
    ax.text(2, v + 0.008, name, color=c, fontsize=8)
ax.set_xlabel("lifetime episode"); ax.set_ylabel("reward (15-ep rolling mean)")
ax.set_title("CCD: consolidation climbs ONLY with the persistent time-clock; without it, it churns flat", fontsize=10.5, fontweight="bold")
ax.legend(frameon=False, loc="lower right", fontsize=9); ax.set_ylim(0, 0.92)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.text(0.995, 0.006, "Qwen3.5-9B + LoRA · start=full-history ET(N3000) · 100 W2W goals · prequential (reward before consolidating) · "
         "persistent-RoPE clock G", ha="right", va="bottom", fontsize=6.5, color="#666")
fig.savefig(f"{R}/ccd_hillclimb.png", dpi=130, bbox_inches="tight")
print("saved ccd_hillclimb.png")
