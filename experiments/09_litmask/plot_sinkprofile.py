"""Sink ratio vs context length at the generation token, across training regimes.
Reads sinkprof_<m>.npz (p0/reg/band per query position, mean over layers/heads/instances).
Run: python plot_sinkprofile.py <npzdir> [--out sink_vs_depth.png]
"""
import numpy as np, argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("npzdir")
ap.add_argument("--out", default="sink_vs_depth.png")
ap.add_argument("--smooth", type=int, default=33)
a = ap.parse_args()

NAME = {"a": "a triangle", "b": "b Gemma/SWAT", "c": "c SWAA", "f": "f sliding-history",
        "g": "g +warmup", "h": "h regs (abs)", "i": "i regs (RIDING)", "j": "j triangle+regs", "k": "k b+sink token", "l": "l b+sink SCALAR", "m": "m b+sink PREFIX", "n": "n b+RIDING token", "o": "o b+RIDING prefix", "p": "p a+sink SCALAR", "q": "q a+sink PREFIX"}
COL = {"a": "#888888", "b": "#2e6db4", "c": "#4d4d4d", "f": "#1b9e77", "g": "#e6ab02",
       "h": "#d95f02", "i": "#c2185b", "j": "#7b1fa2", "k": "#00838f", "l": "#558b2f", "m": "#5d4037", "n": "#e91e63", "o": "#3949ab", "p": "#827717", "q": "#00695c"}

def sm(x, k):
    if k <= 1: return x
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="valid")

D = {}
for m in NAME:
    p = os.path.join(a.npzdir, "sinkprof_%s.npz" % m)
    if os.path.exists(p): D[m] = np.load(p)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
for m, z in D.items():
    total = z["p0"] + z["reg"] + z["band"]
    y = sm(total, a.smooth); x = np.arange(len(y)) + a.smooth // 2
    ax1.plot(x, y, color=COL[m], lw=1.9, ls="--" if m in "ac" else "-", label=NAME[m])
ax1.axvline(64, color="black", lw=0.8, ls=":", alpha=0.6)
ax1.text(70, ax1.get_ylim()[1] * 0.02, "W=64", fontsize=8)
ax1.set_xlabel("query position (context length at the generation token)")
ax1.set_ylabel("parked attention ratio (p0 + registers + persistent cols)")
ax1.set_title("total sink ratio vs context length"); ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

for m in ("a", "c"):
    if m in D:
        y = sm(D[m]["p0"], a.smooth); x = np.arange(len(y)) + a.smooth // 2
        ax2.plot(x, y, color=COL[m], lw=1.9, ls="--", label=NAME[m] + ": pos-0 mass")
for m in ("h", "i", "j", "k", "l", "m", "n", "o", "p", "q"):
    if m in D:
        y = sm(D[m]["reg"], a.smooth); x = np.arange(len(y)) + a.smooth // 2
        ax2.plot(x, y, color=COL[m], lw=2.2, label=NAME[m] + ": register mass")
ax2.set_xlabel("query position (context length at the generation token)")
ax2.set_ylabel("anchor-class attention ratio")
ax2.set_title("anchor mass vs context length (drift test)")
ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(a.out, dpi=150, facecolor="white")
print("wrote", a.out)
