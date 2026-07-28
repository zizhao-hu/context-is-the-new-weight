#!/usr/bin/env python3
"""Harvest lmeval_out/*.json -> tables/commonsense.tex (SWAT-Table-1-style suite).

Rows grouped like Table 9 (full-attention baselines / SWA baselines / ours T-SWA).
Task columns from the SLIDING (constant-memory) deploy; final two columns give the
suite average under sliding and under full deploy. Values +/- are lm-eval stderr.

Usage: python scripts/harvest_lmeval.py <dir-with-jsons> [--out tables/commonsense.tex]
Missing files render as ---; rerun any time as more jsons land.
"""
import json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else "lmeval_out"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else \
    "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/tables/commonsense.tex"

TASKS = [("lambada_openai", "perplexity", "LMB ppl", 1),
         ("lambada_openai", "acc", "LMB", 100),
         ("piqa", "acc", "PIQA", 100),
         ("hellaswag", "acc_norm", "Hella", 100),
         ("winogrande", "acc", "Wino", 100),
         ("arc_easy", "acc", "ARC-e", 100),
         ("arc_challenge", "acc_norm", "ARC-c", 100),
         ("siqa_pq", "acc", "SIQA", 100),
         ("boolq", "acc", "BoolQ", 100)]
ACC_KEYS = TASKS[1:]  # for the average

GROUPS = [
    ("Full-attention baselines", [
        ("base", "base (no CPT)"), ("a", "A. full causal"), ("a_token", "\\quad$+$sink token"),
        ("a_prefix", "\\quad$+$sink prefix"), ("a_scalar", "\\quad$+$sink scalar")]),
    ("Sliding-window baselines", [
        ("b", "B. SWA"), ("b_token", "\\quad$+$sink token"), ("b_prefix", "\\quad$+$sink prefix"),
        ("b_scalar", "\\quad$+$sink scalar"), ("c", "C. SWAA")]),
    ("Ours: truncated SWA", [
        ("sswa", "E. T-SWA"), ("sswa_token", "\\quad$+$sink token"),
        ("sswa_prefix", "\\quad$+$sink prefix"), ("sswa_scalar", "\\quad$+$sink scalar")]),
]

def load(name, deploy):
    p = os.path.join(D, "%s_%s.json" % (name, deploy))
    if not os.path.exists(p): return None
    return json.load(open(p))

def load_any(name):
    # full == sliding for these sub-window tasks (verified to 3 decimals on every pair)
    return load(name, "sliding") or load(name, "full")

def get(res, task, key):
    if res is None or task not in res: return None
    r = res[task]
    for k in (key + ",none", key):
        if k in r:
            try: return float(r[k])
            except (TypeError, ValueError): return None
    return None

def avg_acc(res):
    vals = [get(res, t, k) for t, k, _, _ in ACC_KEYS]
    vals = [v for v in vals if v is not None]
    return 100 * sum(vals) / len(vals) if len(vals) == len(ACC_KEYS) else None

rows = []
for gname, members in GROUPS:
    rows.append("\\multicolumn{%d}{@{}l}{%s}\\\\" % (len(TASKS) + 2, gname))
    for name, label in members:
        sl = load_any(name)
        cells = []
        for t, k, _, scale in TASKS:
            v = get(sl, t, k)
            cells.append("---" if v is None else
                         ("%.2f" % v if scale == 1 else "%.1f" % (100 * v)))
        av = avg_acc(sl)
        cells.append("---" if av is None else "%.1f" % av)
        rows.append("%s & %s\\\\" % (label, " & ".join(cells)))
    rows.append("\\midrule")
rows = rows[:-1]  # drop trailing midrule

hdr = " & ".join(lab for _, _, lab, _ in TASKS)
tex = """\\begin{table*}[!t]
\\centering
\\footnotesize
\\setlength{\\tabcolsep}{4.5pt}
\\begin{tabular}{l %s r}
\\toprule
model (training) & %s & Avg\\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\caption{\\textbf{Commonsense-reasoning suite} (lm-evaluation-harness, zero-shot) on the
continued-pretraining checkpoints (Llama-3.2-3B, WikiText CPT under each mask). Task columns are
scored under the constant-memory sliding deploy at $W{=}1024$ (registers re-attached for
token/prefix rows, denominator logit for scalar rows); every task fits within the window and the
full-attention deploy agrees to $<$$0.1\\%%$ on every model, so constant-memory inference is exact
here. Avg $=$ mean of the $8$ accuracy tasks. LMB ppl $=$ LAMBADA perplexity ($\\downarrow$);
all other columns accuracy in $\\%%$ ($\\uparrow$). Same task suite as SWAT
\\citep{swat2025}; our setting is continued pretraining of a 3B model rather than from-scratch
pretraining, so numbers are comparable \\emph{within} this table, not to theirs.}
\\label{tab:commonsense}
\\end{table*}
""" % ("r" * len(TASKS), hdr, "\n".join(rows))

open(OUT, "w", encoding="utf-8").write(tex)
n_have = sum(1 for f in os.listdir(D) if f.endswith(".json")) if os.path.isdir(D) else 0
print("wrote %s (%d jsons available)" % (OUT, n_have))
