"""Compact publication results figure: 2x2 panel grid, minimal text, config block bottom-right."""
import json, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

R = "/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/eval_out"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

def sr(path, key=None):
    try:
        d = json.load(open(path)); v = d.get(key) if key else list(d.values())[0]
        eps = v.get("episodes", [])
        return 100*sum(1 for e in eps if e["reward"] >= 1.0)/len(eps) if eps else None
    except Exception:
        return None

# ---- scaling-curve data (read from cells) ----
NS = [150,300,500,1000,2000,3000,5000,8000,10000,15000,20000]
def curve(t):
    xs, ys = [], []
    for n in NS:
        s = sr(f"{R}/rollout_curve_{t}_N{n}.json", f"curve_{t}_N{n}")
        if s is not None: xs.append(n); ys.append(s)
    return xs, ys

C = {"#2e7d32": "Full-history ET", "#888": "Obs→Act", "#e8a13a": "Obs→React"}
fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.2))

# (0,0) prompted reasoning vs scale
sizes = ["9B", "27B", "540B*"]
nan = float("nan")
ax[0,0].plot(sizes, [43,50,30], "o-", color="#1f77b4", lw=2, label="Act (prompt)")
ax[0,0].plot(sizes, [10,22,40], "s--", color="#d62728", lw=2, label="ReAct (prompt)")
ax[0,0].plot(sizes, [36,39,nan], "^-", color="#2e7d32", lw=2, label="ET (trained)")
ax[0,0].set_title("Reasoning: prompted vs trained vs scale")
ax[0,0].set_ylabel("WebShop SR (%)"); ax[0,0].set_ylim(0,60); ax[0,0].legend(frameon=False, fontsize=7.5)
ax[0,0].axvspan(1.5, 2.0, color="grey", alpha=0.06)

# (0,1) trained vs prompted @ 9B
labels = ["Act\n(prompt)", "ReAct\n(prompt)", "ET\n(trained)"]
bars = ax[0,1].bar(labels, [43,10,36], color=["#1f77b4","#d62728","#2e7d32"])
ax[0,1].set_title("Training recovers reasoning (9B)")
ax[0,1].set_ylabel("WebShop SR (%)"); ax[0,1].set_ylim(0,55)
for b,v in zip(bars,[43,10,36]): ax[0,1].text(b.get_x()+b.get_width()/2, v+1, str(v), ha="center", fontsize=8)

# (1,0) full-history scaling vs N
for col,t in [("#888","obsact"),("#e8a13a","obsreact"),("#2e7d32","history")]:
    xs, ys = curve(t)
    ax[1,0].plot(xs, ys, "o-", color=col, lw=1.8, ms=3, label={"obsact":"Obs→Act","obsreact":"Obs→React","history":"Full-hist ET"}[t])
ax[1,0].set_xscale("log"); ax[1,0].set_title("Full-history is the axis that scales")
ax[1,0].set_xlabel("train trajectories N"); ax[1,0].set_ylabel("WebShop SR (%)")
ax[1,0].set_ylim(0,45); ax[1,0].legend(frameon=False, fontsize=7.5)

# (1,1) ALFWorld per-task: Qwen vs paper-540B
tasks = ["Pick","Clean","Heat","Cool","Look","Pick2","All"]
qwen = [0,4,5,7,100,0,17.6]; palm = [92,58,96,86,78,41,71]
x = range(len(tasks)); w = 0.4
ax[1,1].bar([i-w/2 for i in x], qwen, w, color="#2e7d32", label="Qwen 9B")
ax[1,1].bar([i+w/2 for i in x], palm, w, color="#bbb", label="PaLM 540B*")
ax[1,1].set_xticks(list(x)); ax[1,1].set_xticklabels(tasks, fontsize=7.5)
ax[1,1].set_title("ALFWorld per-task SR"); ax[1,1].set_ylabel("SR (%)"); ax[1,1].set_ylim(0,105)
ax[1,1].legend(frameon=False, fontsize=7.5)

fig.tight_layout(rect=[0, 0.045, 1, 1])
fig.text(0.995, 0.008,
         "Base: Qwen3.5-9B / Qwen3.6-27B  |  WebShop & ALFWorld, 100 held-out sessions  |  "
         "SR = % reward=1.0  |  ReAct: greedy, 15-step, few-shot  |  *540B = PaLM (Yao+ 2022)",
         ha="right", va="bottom", fontsize=6.5, color="#666")
fig.savefig(f"{R}/results_fig.png", bbox_inches="tight")
print("saved results_fig.png")
