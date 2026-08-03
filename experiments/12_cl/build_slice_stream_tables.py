#!/usr/bin/env python3
"""Build the slice/stream CL tables from the naive-group rerun logs (cls_* jobs).

Usage: build_slice_stream_tables.py <logdir> <cl_main.tex> <cl_stream.tex>

cl_main.tex   main-text Table: 7 masks x three regimes (trained slice q>=W,
              untrained short slice q<W, streaming far slice q>=L at the W budget)
cl_stream.tex appendix detail: per-deploy-policy streaming finals + forgetting
"""
import glob, math, re, sys

LOGDIR, OUT_MAIN, OUT_STREAM = sys.argv[1], sys.argv[2], sys.argv[3]
OUT_REGIMES = sys.argv[4] if len(sys.argv) > 4 else OUT_MAIN.replace("cl_main", "cl_regimes")
STAGE_OF = {"wikitext": 1, "gsm8k": 2, "tofu": 3, "arc": 4}
MASKS = ["a", "b", "bs", "c", "d", "t", "ts"]
NAME = {"a": "A.\\ full causal", "b": "B.\\ SWA", "bs": "B $+$ sink prefix",
        "c": "C.\\ SWAA", "d": "D.\\ Transformer-XL", "t": "E.\\ T-SWA",
        "ts": "E $+$ sink prefix"}
NATIVE = {"a": "sllm", "b": "slide", "bs": "prefix", "c": "sllm", "d": "txl",
          "t": "slide", "ts": "prefix"}

def parse(f):
    ev = {}
    for ln in open(f):
        m = re.match(r"CLEVAL(SHORT)? stage=(\d) task=(\w+) ppl=([\d.]+) sem=([\d.]+)", ln)
        if m:
            ev[("short" if m[1] else "long", int(m[2]), m[3])] = (float(m[4]), float(m[5]))
        m = re.match(r"STREAMEVAL stage=(\d) task=(\w+) policy=(\w+) all=([\d.]+) asem=([\d.]+) "
                     r"far=([\d.]+) fsem=([\d.]+)", ln)
        if m:
            ev[("far", m[3], int(m[1]), m[2])] = (float(m[6]), float(m[7]))
    return ev

runs = {}
for mask in MASKS:
    fs = glob.glob("%s/cl_cls_%s_naive_*.log" % (LOGDIR, mask))
    assert len(fs) == 1, (mask, fs)
    runs[mask] = parse(fs[0])

def learn(ev, key, tasks_):
    """mean best post-learning ppl (peak mastery), with propagated SEM"""
    bs_ = [min((ev[key + (s, t)] for s in range(STAGE_OF[t], 5)), key=lambda x: x[0])
           for t in tasks_]
    n = len(bs_)
    return sum(v for v, _ in bs_) / n, math.sqrt(sum(sm ** 2 for _, sm in bs_)) / n

def forget(ev, key, tasks_):
    """mean rise from post-learning best, with propagated SEM (0 when best = final)"""
    vals, sems = [], []
    for t in tasks_:
        series = [(ev[key + (s, t)]) for s in range(STAGE_OF[t], 5)]
        best = min(range(len(series)), key=lambda i: series[i][0])
        fin = series[-1]
        vals.append(fin[0] - series[best][0])
        sems.append(0.0 if best == len(series) - 1
                    else math.sqrt(fin[1] ** 2 + series[best][1] ** 2))
    n = len(vals)
    return sum(vals) / n, math.sqrt(sum(s ** 2 for s in sems)) / n

def pm(v, s, prec=2, sign=False):
    f = "%+." + str(prec) + "f" if sign else "%." + str(prec) + "f"
    return (f % v) + ("{\\tiny$\\pm$%.*f}" % (max(prec - 1, 1), s))

# ---------------- main table: the 2x2 recipe ablation ----------------
MAIN = ["a", "b", "t", "bs", "ts"]
MAIN_NAME = {"a": "A.\\ full causal (reference)", "b": "B.\\ SWA (sliding base)",
             "t": "E.\\ T-SWA ($+$truncated loss)", "bs": "B $+$ sink prefix ($+$sink)",
             "ts": "E $+$ sink prefix (both)"}
rows, cols = {}, {m: {} for m in MASKS}
for m in MASKS:
    ev = runs[m]
    cols[m]["f_long"] = forget(ev, ("long",), ("wikitext", "gsm8k", "tofu"))
    cols[m]["learn"] = learn(ev, ("long",), ("wikitext", "gsm8k", "arc"))
    if m not in ("bs", "ts"):    # prefix registers occupy the q<W deploy: slice excluded
        cols[m]["f_short"] = forget(ev, ("short",), ("wikitext", "gsm8k", "tofu"))
        b, fb = ev[("short", 0, "fineweb")], ev[("short", 4, "fineweb")]
        cols[m]["drift"] = (fb[0] - b[0], math.sqrt(b[1] ** 2 + fb[1] ** 2))
    pol = NATIVE[m]
    cols[m]["f_far"] = forget(ev, ("far", pol), ("wikitext", "gsm8k"))
    cols[m]["fw_far"] = ev[("far", pol, 4, "fineweb")]

best = {}
for c in ("f_long", "learn", "f_short", "drift", "f_far", "fw_far"):
    prec = 1 if c in ("drift", "fw_far", "learn") else 2
    have = [(round(cols[m][c][0], prec), m) for m in MAIN if c in cols[m]]
    lo = min(v for v, _ in have)
    best[c] = {m for v, m in have if v == lo}

def cell(m, c, prec=2, sign=False):
    if c not in cols[m]:
        return "---"
    s = pm(*cols[m][c], prec=prec, sign=sign)
    return "\\textbf{%s}" % s if m in best[c] else s

L = []
L.append("\\begin{table*}[t]")
L.append("\\centering")
L.append("\\setlength{\\abovecaptionskip}{4pt}")
L.append("\\scriptsize")
L.append("\\setlength{\\tabcolsep}{2.6pt}")
L.append("\\renewcommand{\\arraystretch}{0.92}")
L.append("\\begin{tabular}{@{}l rr rr rr@{}}")
L.append("\\toprule")
L.append(" & \\multicolumn{2}{c}{untrained short} & \\multicolumn{2}{c}{trained} & "
         "\\multicolumn{2}{c}{streaming far}\\\\")
L.append(" & \\multicolumn{2}{c}{$q < W$} & \\multicolumn{2}{c}{$q \\ge W$} & "
         "\\multicolumn{2}{c}{$4096$ tok, KV $\\le W$}\\\\")
L.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}")
L.append("training & $\\Delta$base$\\downarrow$ & F$\\downarrow$ & learn$\\downarrow$ & "
         "F$\\downarrow$ & F$\\downarrow$ & general$\\downarrow$\\\\")
L.append("\\midrule")
for m in MAIN:
    L.append("%s & %s & %s & %s & %s & %s & %s\\\\" %
             (MAIN_NAME[m], cell(m, "drift", prec=1, sign=True), cell(m, "f_short"),
              cell(m, "learn", prec=1), cell(m, "f_long"),
              cell(m, "f_far"), cell(m, "fw_far", prec=1)))
L.append("\\bottomrule")
L.append("\\end{tabular}")
L.append("\\caption{\\textbf{Continual learning under the three deploy regimes} "
         "(the recipe's two components ablated on the sliding base; naive sequential "
         "training, equal supervised tokens). F, forgetting: rise of "
         "held-out task ppl from its post-learning best, mean over the first three tasks "
         "(streaming: first two; the TOFU stream is shorter than one $4096$-token "
         "sequence). Short slice: the rows the truncated loss never supervises, where the "
         "deploys of the plain masks coincide (shared base $28.8$ on FineWeb); "
         "$\\Delta$base, shift of never-trained FineWeb ppl from the base model. Prefix "
         "rows are excluded there: their registers occupy exactly this deploy, so the "
         "slice no longer isolates the loss rule. learn: best post-learning ppl, mean "
         "over the three natural tasks; TOFU, $360$-row verbatim memorization, is "
         "excluded there (equal-budget fit measures overfit speed; per task in "
         "Tab.~\\ref{tab:cltasks}). streaming general: "
         "final FineWeb ppl past the trained length ($q \\ge L$) at the $W$ KV budget, "
         "each design deploying natively; A must stream via StreamingLLM pinning, since "
         "plain sliding collapses it (FineWeb $154$, F $+29.9$). Bold, best per column. "
         "The literature baselines C (SWAA) and D (Transformer-XL) are in "
         "Tabs.~\\ref{tab:cl} and~\\ref{tab:clstream}; D reaches F $0.94$ trained and "
         "$1.60$ streaming at a doubled ($2W$) KV budget. Methods, backward transfer, "
         "and per-policy streaming detail: App.~\\ref{app:cltasks}.}")
L.append("\\label{tab:clregimes}")
L.append("\\end{table*}")
open(OUT_REGIMES, "w").write("\n".join(L) + "\n")

# ---------------- main table: the recipe against the two canonical baselines ----------------
CPT = [("a", "A.\\ full causal"), ("b", "B.\\ SWA"), ("ts", "T-SWA ($+$sink prefix)")]
cbest = {}
for c in ("learn", "f_long", "f_far", "fw_far"):
    prec = 1 if c in ("fw_far", "learn") else 2
    have = [(round(cols[m][c][0], prec), m) for m, _ in CPT]
    lo = min(v for v, _ in have)
    cbest[c] = {m for v, m in have if v == lo}
def ccell(m, c, prec=2):
    t = pm(*cols[m][c], prec=prec)
    return "\\textbf{%s}" % t if m in cbest[c] else t
M = []
M.append("\\begin{table}[t]")
M.append("\\centering")
M.append("\\setlength{\\abovecaptionskip}{4pt}")
M.append("\\scriptsize")
M.append("\\setlength{\\tabcolsep}{3.4pt}")
M.append("\\renewcommand{\\arraystretch}{0.95}")
M.append("\\begin{tabular}{@{}l rr rr@{}}")
M.append("\\toprule")
M.append(" & \\multicolumn{2}{c}{trained $q \\ge W$} & \\multicolumn{2}{c}{streaming far}\\\\")
M.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
M.append("training & learn$\\downarrow$ & F$\\downarrow$ & F$\\downarrow$ & general$\\downarrow$\\\\")
M.append("\\midrule")
for m, nm in CPT:
    M.append("%s & %s & %s & %s & %s\\\\" %
             (nm, ccell(m, "learn", prec=1), ccell(m, "f_long"),
              ccell(m, "f_far"), ccell(m, "fw_far", prec=1)))
M.append("\\bottomrule")
M.append("\\end{tabular}")
M.append("\\caption{\\textbf{Continual learning at the deploy budget}: full causal, SWA, and "
         "T-SWA with its sink prefix, the recipe default (naive sequential training, equal "
         "supervised tokens). learn: best post-learning ppl over the three natural tasks. "
         "F: forgetting, rise of held-out task ppl from its post-learning best, on the "
         "trained slice ($q \\ge W$, first three tasks) and on $4096$-token streams past "
         "the trained length at the $W$ KV budget (first two; the TOFU stream is shorter "
         "than one sequence), where A must stream via StreamingLLM pinning since plain "
         "sliding collapses it. general: final streaming FineWeb ppl. Bold, best per "
         "column. The component ablation with the untrained-short slice, the literature "
         "baselines, methods, and per-task detail are in App.~\\ref{app:cltasks}.}")
M.append("\\label{tab:clmain}")
M.append("\\end{table}")
open(OUT_MAIN, "w").write("\n".join(M) + "\n")

# ---------------- appendix streaming detail ----------------
ROWS = [("a", "full", "A, full attention (unbounded)"),
        ("a", "slide", "A, sliding $W$"),
        ("a", "sllm", "A, StreamingLLM $4{+}$recent"),
        ("b", "slide", "B, sliding $W$ (native)"),
        ("bs", "prefix", "B $+$ sink, prefix $+$ recent"),
        ("c", "sllm", "C, pinned $4{+}$recent (native)"),
        ("d", "txl", "D, segment recurrence ($2W$)"),
        ("t", "slide", "E, sliding $W$ (native)"),
        ("ts", "prefix", "E $+$ sink, prefix $+$ recent")]
TASKS = ("wikitext", "gsm8k", "arc", "fineweb")
S = []
S.append("\\begin{table*}[t]")
S.append("\\centering")
S.append("\\setlength{\\abovecaptionskip}{4pt}")
S.append("\\scriptsize")
S.append("\\setlength{\\tabcolsep}{2.6pt}")
S.append("\\renewcommand{\\arraystretch}{0.92}")
S.append("\\begin{tabular}{@{}l rrrr r@{}}")
S.append("\\toprule")
S.append("deploy & \\multicolumn{1}{c}{wikitext} & \\multicolumn{1}{c}{gsm8k} & "
         "\\multicolumn{1}{c}{arc} & \\multicolumn{1}{c}{fineweb} & F$\\downarrow$\\\\")
S.append("\\midrule")
for m, pol, label in ROWS:
    ev = runs[m]
    cs = [pm(*ev[("far", pol, 4, t)], prec=1) for t in TASKS]
    fv, fs = forget(ev, ("far", pol), ("wikitext", "gsm8k"))
    S.append("%s & %s & %s\\\\" % (label, " & ".join(cs), pm(fv, fs)))
S.append("\\bottomrule")
S.append("\\end{tabular}")
S.append("\\caption{Streaming detail behind Tab.~\\ref{tab:clmain}: final (stage-$4$) "
         "far-slice ppl ($4096$-token streams, scored $q \\ge L{=}1024$) per deploy "
         "policy, and streaming forgetting F (first two tasks). Cache never exceeds "
         "$W{=}256$ entries except D ($2W$); A's unbounded row is the reference its "
         "bounded deploys chase. Base-model anchors on FineWeb: full $20.0$, sliding "
         "$85.9$, StreamingLLM $22.1$.}")
S.append("\\label{tab:clstream}")
S.append("\\end{table*}")
open(OUT_STREAM, "w").write("\n".join(S) + "\n")
print("wrote", OUT_MAIN, "and", OUT_STREAM)
