#!/usr/bin/env python3
"""Rebuild tables/toydeploy.tex as the FAIR grid: W=1024 (CPT) / W=128 (toy), C matched,
one uniform ppl>C (chunked 30k stream at budget, registers attached), consistent SEMs.

CPT values are parsed from fair_dump.txt (grep dump of the campaign logs); toy values are
hardcoded below with per-value provenance (job/log) since they span many small runs.
Conventions:
  ppl<C  = full-attention deploy, deep bin (register-aware for token/prefix rows)
  ppl>C  = 30k chunked stream at budget W; sink rows use their trained sink; sink-free rows
           use the better of plain sliding / StreamingLLM (stated per-row via ^s marker)
  cc (E/F/G) rows: ppl<C = --- (train only full-window queries)
Usage: python scripts/build_fair_table.py <fair_dump.txt>
"""
import os, re, sys

DUMP = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/tables/toydeploy.tex")
d = open(DUMP, encoding="utf-8").read()

def f(pat, cast=float):
    m = re.search(pat, d)
    return cast(m.group(1)) if m else None

# ---------------- CPT: ppl<C (full deploy deep) ----------------
# b-family from W=1024 RESULT lines; token/prefix from register-aware causal (validated)
# pplS awaits the 30k deep-stream eval (job 5165092); the short-eval 9.97 is NOT comparable
# to the 30k column and must not be printed. Inject via CLI: base_ppls=<val>.
# The no-CPT row must be the model the CPT rows actually start from: Llama-3.2-3B-Instruct
# (cpt_masks.py default). The pretrained Llama-3.2-3B sits at 6.50/7.72 but is a different
# model, so putting it here made every CPT row look like a regression (job 5163378 = Instruct,
# steps=0: full deep 9.29, HotpotQA full .560/.587, streaming .430/.500).
BASE = {"pplC": 9.29, "pplC_sem": 0.7, "pplS": None, "hp": (0.560, 0.587), "hp_stream": (0.430, 0.500)}
for arg in sys.argv[2:]:
    if arg.startswith("base_ppls="): BASE["pplS"] = float(arg.split("=")[1])
for arg in sys.argv[2:]:
    if arg.startswith("base_hs="):
        BASE["hp_stream"] = tuple(float(v) for v in arg.split("=")[1].split(","))

pplC = {
    # FRESH a-family (w1k_afresh jobs 5164970-73, same 4883-step recipe as the b family)
    "a": 8.93, "a_tok": 10.14, "a_scal": 8.90, "a_pref": 11.49, "a_2p0": 10.10,
    "b":       f(r"w1k_b_\d+\|RESULT cpt mask=b .*full\(avg [\d.]+ deep ([\d.]+)\)"),
    "b_tok":   10.44,  # RESULT_REGDEPLOY reg=token/causal (job 5153822)
    "b_pref":  10.44,  # RESULT_REGDEPLOY reg=prefix/causal (job 5153822)
    "b_rtok":  f(r"w1k_b_ridtok_\d+\|RESULT cpt .*full\(avg [\d.]+ deep ([\d.]+)\)"),
    "b_scal":  f(r"w1k_b_scalar_\d+\|RESULT cpt .*full\(avg [\d.]+ deep ([\d.]+)\)"),
    "b_rpref": f(r"w1k_b_ridpref_\d+\|RESULT cpt .*full\(avg [\d.]+ deep ([\d.]+)\)"),
    "c":       f(r"w1k_c_\d+\|RESULT cpt .*full\(avg [\d.]+ deep ([\d.]+)\)"),
    "d":       f(r"w1k_d_txl_\d+\|RESULT cpt .*full\(avg [\d.]+ deep ([\d.]+)\)"),
}
# SEM for ppl<C (kind=full deep_sem) where available
pplC_sem = {
    "a": 1.2, "a_tok": 1.4, "a_scal": 1.5, "a_pref": 1.7, "a_2p0": 1.0,
    "b":       f(r"w1k_b_\d+\|SEM_DEPLOY kind=full n=\d+ avg_sem=[\d.]+ deep_sem=([\d.]+)"),
    "b_tok": 1.1, "b_pref": 1.1,
    "b_rtok":  f(r"w1k_b_ridtok_\d+\|SEM_DEPLOY kind=full .*deep_sem=([\d.]+)"),
    "b_scal":  f(r"w1k_b_scalar_\d+\|SEM_DEPLOY kind=full .*deep_sem=([\d.]+)"),
    "b_rpref": f(r"w1k_b_ridpref_\d+\|SEM_DEPLOY kind=full .*deep_sem=([\d.]+)"),
    "c":       f(r"w1k_c_\d+\|SEM_DEPLOY kind=full .*deep_sem=([\d.]+)"),
    "d":       f(r"w1k_d_txl_\d+\|SEM_DEPLOY kind=full .*deep_sem=([\d.]+)"),
}

# ---------------- CPT: ppl>C (30k stream @1024) ----------------
def ds(model, kind):   # DSTREAM deep
    return f(r"DSTREAM model=%s \| .*%s\(avg [\d.]+ deep ([\d.]+) @30k\)" % (re.escape(model), kind))
def rgv(tag):          # RESULT_REGDEPLOY sliding deep
    return f(r"RESULT_REGDEPLOY mask=\S+ tag=%s reg=\S+/sliding \| avg [\d.]+ deep ([\d.]+)" % re.escape(tag))

# E rows under a full-attention deploy, deep bin: measured, but out of distribution
# (trained at W=1024, deployed with 2048 of reach). From the bin0 and ccx runs.
E_FULL = {"e": 13.87, "f_tok": 13.38, "f_pref": 104.58, "f_scal": 9.65,
          "g_rtok": 9.52, "g_rpref": 9.78}

pplS = {  # value, marker: s=StreamingLLM, w=own sliding, r=registers
    "a":      (ds("a", "streamingLLM"), "s"),
    "a_tok":  (rgv("j_iid"), "r"),
    "a_scal": (ds("a_scal", "sliding"), "w"),
    "a_pref": (rgv("j_iid_pl"), "r"),
    "a_2p0":  (ds("x2a_2p0", "sliding"), "w"),
    "b":      (min(ds("b", "sliding"), ds("b", "streamingLLM")), "s" if ds("b", "streamingLLM") < ds("b", "sliding") else "w"),
    "b_tok":  (rgv("h_iid"), "r"),
    "b_rtok": (min(ds("n_iid", "sliding"), ds("n_iid", "streamingLLM")), "w"),
    "b_scal": (ds("b_scal", "sliding"), "w"),
    "b_pref": (rgv("h_iid_pl"), "r"),
    "b_rpref":(min(ds("o_iid", "sliding"), ds("o_iid", "streamingLLM")), "w"),
    "c":      (ds("c", "streamingLLM"), "s"),
    "d":      (ds("e_B", "streamingLLM"), "s"),
    "e":      (ds("b_cc_w1024", "sliding"), "w"),
    "f_tok":  (rgv("h_iid_cc_w1024"), "r"),
    "f_pref": (rgv("h_iid_cc_w1024_pl"), "r"),
    "f_scal": (ds("b_scal_cc_w1024", "sliding"), "w"),
    "g_rtok": (min(ds("n_iid_cc_w1024", "sliding"), ds("n_iid_cc_w1024", "streamingLLM")), "w"),
    "g_rpref":(min(ds("o_iid_cc_w1024", "sliding"), ds("o_iid_cc_w1024", "streamingLLM")), "w"),
}

# ---------------- CPT: task (HotpotQA F1/Acc, full + windowed@1024) ----------------
def hp(model, deploy):
    m = re.search(r"HOTPOT model=%s deploy=%s W=\d+ \| F1 ([\d.]+) \| EM [\d.]+ \| Acc ([\d.]+)"
                  % (re.escape(model), deploy), d)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)

TASK = {
    "a": ("a", "streaming"), "a_tok": ("j_iid", "windowed"), "a_scal": ("a_scal", "windowed"),
    "a_pref": ("j_iid_pl", "windowed"), "a_2p0": ("x2a_2p0", "windowed"),
    "b": ("b", "windowed"), "b_tok": ("h_iid", "windowed"), "b_rtok": ("n_iid", "windowed"),
    "b_scal": ("b_scal", "windowed"), "b_pref": ("h_iid_pl", "windowed"), "b_rpref": ("o_iid", "windowed"),
    "c": ("c", "streaming"), "d": ("e_B", "windowed"),
    "e": ("b_cc_w1024", "windowed"), "f_tok": ("h_iid_cc_w1024", "windowed"),
    "f_pref": ("h_iid_cc_w1024_pl", "windowed"), "f_scal": ("b_scal_cc_w1024", "windowed"),
    "g_rtok": ("n_iid_cc_w1024", "windowed"), "g_rpref": ("o_iid_cc_w1024", "windowed"),
}
task = {}
for k, (mdl, dep) in TASK.items():
    task[k] = hp(mdl, "full") + hp(mdl, dep)   # (F1f, Accf, F1s, Accs)

# ---------------- TOY (W=128, C=256): hardcoded w/ provenance ----------------
# in-context (incx Lm=256; regs=True where registers exist) | stream (30k deep, sem)
toy = {  # key: (pplC, pplS, sem, marker)
    "a":      (33.0, 41.3, 1.7, "s"),   # incx historical / toy128 job streaming
    "a_tok":  (33.3, 39.7, 1.6, "r"),   # toy128b j DEPLOY_EXTRA
    "a_scal": (32.8, 39.7, 1.7, "w"),   # toy128 p native sliding
    "a_pref": (32.4, None, None, "r"),  # q pending (job 5160312)
    "a_2p0":  (None, None, None, ""),   # no toy 2-slot model
    "b":      (41.1, 39.2, 1.6, "w"),   # incx128 / b_w128 DEPLOY
    "b_tok":  (48.9, 41.1, 1.7, "r"),   # incx128 regs / bvcc k EXTRA
    "b_rtok": (55.1, 39.3, 1.6, "r"),   # incx128 regs / bvcc n EXTRA
    "b_scal": (55.7, 38.9, 1.6, "w"),   # incx128 regs / bvcc l EXTRA (native scalar)
    "b_pref": (60.3, 38.8, 1.6, "r"),   # incx128 regs / bvcc m EXTRA
    "b_rpref":(58.8, 38.9, 1.6, "r"),   # incx128 regs / bvcc o EXTRA
    "c":      (33.1, 40.6, 1.6, "s"),   # incx128 / w128_c streaming
    "d":      (None, 41.9, 1.7, "s"),   # incx e pending (job 5160313) / w128_d streaming
    "e":      (None, 39.2, 1.6, "w"),   # cc convention: no in-context
    "f_tok":  (None, 40.1, 1.7, "r"),
    "f_pref": (None, 38.6, 1.6, "r"),
    "f_scal": (None, 38.5, 1.6, "w"),
    "g_rtok": (None, 38.8, 1.6, "r"),   # bvcc cc_n EXTRA
    "g_rpref":(None, 38.3, 1.6, "r"),   # bvcc cc_o EXTRA
}
# late-arriving values injected via CLI:  --set a_pref_ppls=XX.X --set d_pplc=YY.Y
for arg in sys.argv[2:]:
    if arg.startswith("--set") or arg.startswith("base_"): continue
    k, v = arg.split("=")
    key, fld = k.rsplit("_", 1)
    t = list(toy[key])
    t[{"pplc": 0, "ppls": 1, "sem": 2}[fld]] = float(v)
    toy[key] = tuple(t)

ROWS = [
    ("base", [
        ("BASE", "base (no CPT)")]),
    ("baselines", [                                   # A-E in order, no internal rule among the baselines
        ("a", "A.\\ full causal"),
        ("a_tok", "\\quad$+$trainable p0 sink token"), ("a_scal", "\\quad$+$trainable p0 sink scalar"),
        ("a_pref", "\\quad$+$trainable p0 sink prefix"),
        ("b", "B.\\ SWA"),
        ("b_tok", "\\quad$+$trainable p0 sink token"), ("b_scal", "\\quad$+$trainable p0 sink scalar"),
        ("b_pref", "\\quad$+$trainable p0 sink prefix"),
        ("b_rtok", "\\quad$+$trainable sliding sink token"),
        ("b_rpref", "\\quad$+$trainable sliding sink prefix"),
        ("c", "C.\\ SWAA (sink$+$window)"),
        ("d", "D.\\ Transformer-XL")]),
    ("ours", [
        ("e", "E.\\ S-SWA"),
        ("f_tok", "\\quad$+$trainable p0 sink token"), ("f_pref", "\\quad$+$trainable p0 sink prefix"),
        ("f_scal", "\\quad$+$trainable p0 sink scalar"),
        ("g_rtok", "\\quad$+$trainable sliding sink token"),
        ("g_rpref", "\\quad$+$trainable sliding sink prefix")]),
]

def fmt(v, dec=2, sem=None):
    if v is None: return "---"
    s = ("%%.%df" % dec) % v
    if sem is not None: s += "{\\tiny$\\pm$%.1f}" % sem
    return s

def fmt_task(v, sem):
    if v is None: return "---"
    return "%.3f{\\tiny$\\pm$%.2f}" % (v, sem)

# base (no CPT): measured by base8 job 5163200 (steps=0, W=1024 budget); toy columns n/a.
# ppl>C uses StreamingLLM (plain sliding collapses to 181.9) -> s marker.
toy["BASE"] = (None, None, None, "")
pplC["BASE"] = BASE["pplC"]; pplC_sem["BASE"] = BASE["pplC_sem"]
pplS["BASE"] = (BASE["pplS"], "s") if BASE["pplS"] is not None else (None, "s")
task["BASE"] = (BASE["hp"][0], BASE["hp"][1], BASE["hp_stream"][0], BASE["hp_stream"][1])

# best-in-column bolding (computed post-hoc below on assembled floats)
body = []
colvals = {i: [] for i in range(8)}
grid = []
for gname, members in ROWS:
    grid.append(("HDR", gname))
    for k, label in members:
        tC, tS, tsem, tmark = toy[k]
        cC, cCs = pplC.get(k), pplC_sem.get(k)
        if k in E_FULL: cC, cCs = E_FULL[k], None
        sV, sM = pplS[k]
        F1f, Accf, F1s, Accs = task[k]
        cells = [(tC, None), (tS, tsem), (cC, cCs), (sV, None),
                 (F1f, 0.03), (Accf, 0.04), (F1s, 0.03), (Accs, 0.04)]
        grid.append((k, label, cells, tmark, sM))
        if k == "BASE": continue          # the no-CPT reference is not in the running for best-in-column
        for i, (v, _) in enumerate(cells):
            if v is not None: colvals[i].append(v)

best = {}
for i, vals in colvals.items():
    if not vals: continue
    best[i] = (min(vals) if i < 4 else max(vals))

for row in grid:
    if row[0] == "HDR":
        if body:                                  # group headings dropped; keep a rule between groups
            body.append("\\midrule")
        continue
    k, label, cells, tmark, sM = row
    out = []
    for i, (v, sem) in enumerate(cells):
        if v is None: out.append("---"); continue
        if i < 2:   s = fmt(v, 1, sem)
        elif i == 2: s = fmt(v, 2, sem)
        elif i == 3: s = "%.2f$^{%s}$" % (v, sM)
        elif i in (4, 6): s = fmt_task(v, sem)
        else: s = "%.2f{\\tiny$\\pm$%.2f}" % (v, sem)
        if abs(v - best[i]) < 1e-9:
            s = "\\textbf{" + s.split("{\\tiny")[0].replace("$^{%s}$" % sM, "") + "}" + \
                ("{\\tiny" + s.split("{\\tiny")[1] if "{\\tiny" in s else "") + \
                ("$^{%s}$" % sM if i == 3 else "")
        out.append(s)
    body.append("%s & %s\\\\" % (label, " & ".join(out)))
body = "\n".join(body).replace("\\\\\n\\multicolumn", "\\\\\n\\midrule\n\\multicolumn")

tex = """\\begin{table*}[!t]
\\centering
\\footnotesize
\\setlength{\\tabcolsep}{5pt}
\\begin{tabular*}{\\textwidth}{@{}l@{\\extracolsep{\\fill}} rr rr rr rr@{}}
\\toprule
 & \\multicolumn{2}{c}{Pretraining} & \\multicolumn{6}{c}{Continued pretraining: Llama-3.2-3B}\\\\
\\cmidrule(lr){2-3}\\cmidrule(lr){4-9}
 & \\multicolumn{2}{c}{\\footnotesize \\textbf{task:} WikiText ($\\downarrow$)} & \\multicolumn{2}{c}{\\footnotesize \\textbf{task:} WikiText ($\\downarrow$)} & \\multicolumn{4}{c}{\\footnotesize \\textbf{knowledge-retain:} HotpotQA, zero-shot ($\\uparrow$)}\\\\
\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}
 & \\multicolumn{1}{c}{\\footnotesize full} & \\multicolumn{1}{c}{\\footnotesize stream} & \\multicolumn{1}{c}{\\footnotesize full} & \\multicolumn{1}{c}{\\footnotesize stream} & \\multicolumn{2}{c}{\\footnotesize full} & \\multicolumn{2}{c}{\\footnotesize stream}\\\\
\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}
mask (training) & \\multicolumn{1}{c}{ppl$_{<C_{\\max}}$} & \\multicolumn{1}{c}{ppl$_{>C_{\\max}}$} & \\multicolumn{1}{c}{ppl$_{<C_{\\max}}$} & \\multicolumn{1}{c}{ppl$_{>C_{\\max}}$} & \\multicolumn{1}{c}{F1} & \\multicolumn{1}{c}{Acc} & \\multicolumn{1}{c}{F1} & \\multicolumn{1}{c}{Acc}\\\\
\\midrule
%s
\\bottomrule
\\end{tabular*}
\\caption{\\textbf{One picture across pretraining, continued pretraining, and a downstream task,
window-matched.} $C_{\\max}$ is the max context length seen in training ($256$ toy, $2048$ CPT). Every row shares it and the deploy
budget ($W{=}128$ toy, $1024$ CPT); the SWA family is retrained at the matched window. Pretraining:
an $8$-layer GPT from scratch on WikiText; CPT: Llama-3.2-3B on WikiText under each mask, scored
zero-shot on HotpotQA ($n{=}150$; F1 $=$ token overlap, Acc $=$ containment). ppl$_{<C_{\\max}}$ is the
full-attention deploy within the training length (register-aware for token/prefix rows);
ppl$_{>C_{\\max}}$ streams $30$k tokens ($30$-bin chunked cache, last bin) at the stated budget; its
superscript marks the deploy used for that row (not an error or significance mark): $^{r}$
sliding with the trained registers re-attached, $^{w}$ plain sliding (own sink), $^{s}$
StreamingLLM (kept first tokens; used where it beats plain sliding for sink-free rows).
S-SWA and its sink variants (E) never score the first $W$ rows of a chunk; their
within-context cells are measured but out of distribution, since a full-attention deploy lets them
reach $2048$ tokens back when they only ever trained at $W{=}1024$, and two of the p0-sink rows
break outright under it. SEM shown as $\\pm$ (toy stream $n{=}128$ segments; task $n{=}150$ questions); toy
ppl$_{<C_{\\max}}$ and CPT ppl$_{>C_{\\max}}$ are single pooled estimates. The base row is Llama-3.2-3B-Instruct with no CPT, the checkpoint every CPT row starts from, at the same deploy budget; it is a reference, not a competitor, so bolding marks the best CPT row only. Remaining asymmetry, stated plainly:
at matched loss tokens ($10$M) the S-SWA rows consume $2\\times$ the data tokens of the SWA rows
($20$M vs $10$M); bold marks the best point estimate per column.}
\\label{tab:toydeploy}
\\end{table*}
""" % body

open(OUT, "w", encoding="utf-8").write(tex)
missing = [k for k in pplS if pplS[k][0] is None] + [k for k in task if task[k][0] is None]
print("wrote", OUT)
print("missing CPT values:", missing or "none")
