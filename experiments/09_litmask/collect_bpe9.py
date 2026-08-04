"""Collect the riding-register comparison: parse bpe8_(a..g) + bpe9_full_(h,i) logs ->
table (ppl + sink decomposition) and a 2-panel figure (streaming curves | sink bars).
Run: python collect_bpe9.py <logdir> [--out ride_compare.png]
"""
import re, sys, argparse, math

ap = argparse.ArgumentParser()
ap.add_argument("logdir")
ap.add_argument("--out", default="ride_compare.png")
a = ap.parse_args()

import glob, os
LOGS = {}
for m in "abcdefg":
    c = sorted(glob.glob(os.path.join(a.logdir, "bpe8_%s.log" % m)))
    if c: LOGS[m] = c[-1]
for m in ("h", "i", "j", "k", "l", "m", "n", "o", "p", "q"):
    c = sorted(glob.glob(os.path.join(a.logdir, "bpe9_full_%s_*.log" % m)))
    if c: LOGS[m] = c[-1]

NAME = {"a": "a triangle (full causal)", "b": "b Gemma/SWAT sliding", "c": "c SWAA (sink+win)",
        "d": "d Longformer", "e": "e Transformer-XL", "f": "f sliding-history",
        "g": "g +warmup prefix", "h": "h +persist regs (abs pos)", "i": "i +RIDING regs (const offset)",
        "j": "j a+sink token", "k": "k b+sink token", "l": "l b+sink SCALAR", "m": "m b+sink PREFIX (per-layer KV)", "n": "n b+RIDING sink token", "o": "o b+RIDING sink prefix", "p": "p a+sink SCALAR", "q": "q a+sink PREFIX"}

def gf(pat, s, cast=float):
    m = re.search(pat, s)
    return cast(m.group(1)) if m else None

R = {}
for m, path in LOGS.items():
    txt = open(path).read()
    d = {}
    dep = re.search(r"DEPLOY mask=%s .*" % m, txt)
    if dep:
        s = dep.group(0)
        for k in ("full_avg", "full_deep", "sliding_avg", "sliding_deep", "streaming_avg", "streaming_deep"):
            d[k] = gf(r"%s (\-?[\d.]+)" % k, s)
            d[k + "_sem"] = gf(r"%s \-?[\d.]+ \(sem ([\d.]+)\)" % k, s)
    ex = re.search(r"DEPLOY_EXTRA mask=%s .*" % m, txt)
    if ex:
        d["regs_avg"] = gf(r"streamingregs_avg ([\d.]+)", ex.group(0))
        d["regs_deep"] = gf(r"streamingregs_deep ([\d.]+)", ex.group(0))
    snk = re.search(r"SINK mask=%s .*" % m, txt)
    if snk:
        s = snk.group(0)
        d["sinkmass"] = gf(r"p0_mean=([\d.]+)", s); d["sinkmax"] = gf(r"p0_max=([\d.]+)", s)
        d["bd"] = gf(r"bd_mean=([\d.]+)", s); d["valppl"] = gf(r"valppl=([\d.]+)", s)
    sw = re.search(r"SWEEP mask=%s .*" % m, txt)
    if sw:
        s = sw.group(0)
        d["distmass"] = gf(r"distmass=([\d.]+)", s); d["distsem"] = gf(r"distsem=([\d.]+)", s)
        d["p0mass"] = gf(r"p0mass=([\d.]+)", s); d["p0sem"] = gf(r"p0sem=([\d.]+)", s)
    for tagpat, key in ((r"CURVE %s(?:_B)? means = \[(.*?)\]" % m, "curve"),
                        (r"CURVE_REGS %s(?:_B)? means = \[(.*?)\]" % m, "curve_regs")):
        c = re.search(tagpat, txt)
        if c: d[key] = [float(x) for x in c.group(1).split(",")]
    R[m] = d

hdr = ("mask", "full_avg", "full_deep", "slide_avg", "slide_deep", "sllm_avg", "sllm_deep",
       "regs_avg", "regs_deep", "valppl")
print("\t".join(hdr))
def f(x, n=3): return ("%.*f" % (n, x)) if x is not None else "-"
for m in "abcdefghijklmnopq":
    if m not in R: continue
    d = R[m]
    print("\t".join([NAME[m], f(d.get("full_avg"), 2), f(d.get("full_deep"), 2),
                     f(d.get("sliding_avg"), 2), f(d.get("sliding_deep"), 2),
                     f(d.get("streaming_avg"), 2), f(d.get("streaming_deep"), 2),
                     f(d.get("regs_avg"), 2), f(d.get("regs_deep"), 2), f(d.get("valppl"), 2)]))

# ------------------------------------------------------------------ figure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
COL = {"a": "#888888", "b": "#2e6db4", "f": "#1b9e77", "g": "#e6ab02", "h": "#d95f02", "i": "#c2185b", "j": "#7b1fa2", "k": "#00838f", "l": "#558b2f", "m": "#5d4037", "n": "#e91e63", "o": "#3949ab", "p": "#827717", "q": "#00695c"}
for m in ("a", "b", "f", "g"):
    if m in R and "curve" in R[m]:
        ax1.plot(range(1, 31), R[m]["curve"], color=COL[m], ls="--" if m == "a" else "-",
                 lw=1.6, alpha=0.75, label=NAME[m])
for m in ("h", "i", "j", "k", "l", "m", "n", "o", "p", "q"):
    if m in R:
        cv = R[m].get("curve_regs") or R[m].get("curve")
        ax1.plot(range(1, 31), cv, color=COL[m], lw=2.4, label=NAME[m] + " [regs on]")
        if "curve" in R[m] and "curve_regs" in R[m]:
            ax1.plot(range(1, 31), R[m]["curve"], color=COL[m], lw=1.0, ls=":", alpha=0.6,
                     label=NAME[m] + " [regs off]")
ax1.set_xlabel("stream position (k tokens)"); ax1.set_ylabel("perplexity (per 1k-token bin)")
ax1.set_title("continuous sliding-cache streaming to 30k")
ax1.legend(fontsize=7.5, ncol=1); ax1.grid(alpha=0.25)

# consistent persistent-column decomposition (DECOMP lines from sinkdecomp_bpe9.py), additive to 1
DEC = {}
for path in glob.glob(os.path.join(a.logdir, "*decomp*.log")):
    for ln in open(path):
        m = re.match(r"DECOMP mask=(\w) scope=allheads p0=([\d.]+) reg=([\d.]+) band=([\d.]+) rest=([\d.]+)", ln)
        if m: DEC[m.group(1)] = [float(x) for x in m.groups()[1:]]
masks = [m for m in "abcfghijklmnopq" if m in DEC] or [m for m in "abfghi" if m in R]
xs = list(range(len(masks)))
if DEC:
    p0 = [DEC[m][0] for m in masks]; reg = [DEC[m][1] for m in masks]
    bd = [DEC[m][2] for m in masks]; rs = [DEC[m][3] for m in masks]
    ax2.bar(xs, p0, 0.6, color="#555555", label="pos-0 sink")
    ax2.bar(xs, reg, 0.6, bottom=p0, color="#c2185b", label="trainable registers")
    b2 = [p + r for p, r in zip(p0, reg)]
    ax2.bar(xs, bd, 0.6, bottom=b2, color="#2e6db4", label="persistent columns (separators)")
    b3 = [b + d for b, d in zip(b2, bd)]
    ax2.bar(xs, rs, 0.6, bottom=b3, color="#dddddd", label="ordinary content")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("deep-query attention decomposition (all heads, sums to 1)")
else:
    reg = [R[m].get("sinkmass") if m in "hi" else 0 for m in masks]
    p0 = [R[m].get("p0mass") or 0 for m in masks]
    dm = [R[m].get("distmass") or 0 for m in masks]
    wd = 0.27
    ax2.bar([x - wd for x in xs], reg, wd, color="#c2185b", label="register mass (deep q)")
    ax2.bar(xs, p0, wd, color="#555555", label="pos-0 mass (early q)")
    ax2.bar([x + wd for x in xs], dm, wd, color="#2e6db4", label="distributed sink mass")
    ax2.set_title("sink decomposition (sink-head)")
ax2.set_xticks(xs); ax2.set_xticklabels([m for m in masks])
ax2.legend(fontsize=8); ax2.grid(alpha=0.25, axis="y")
plt.tight_layout(); plt.savefig(a.out, dpi=150, facecolor="white")
print("wrote", a.out)
