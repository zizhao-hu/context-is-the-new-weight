#!/usr/bin/env python3
"""Harvest LENPPL lines -> tables/lengthppl.tex (SWAT-Table-2-style length sweep, PG-19).

Columns: eval lengths x {full, sliding@1024} deploy. Rows grouped full-attn / SWA / ours.
Usage: python scripts/harvest_lenppl.py <lenppl.txt>
"""
import re, sys

SRC = sys.argv[1]
OUT = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/tables/lengthppl.tex"
LENS = [128, 1024, 4096, 16384]

data = {}   # (name, deploy, L) -> (ppl, sem)
skip = set()
for ln in open(SRC):
    m = re.search(r"LENPPL ckpt=(\S+) deploy=(\S+) L=(\d+) n=\d+ ppl=([0-9.]+) sem=([0-9.]+)", ln)
    if m:
        data[(m.group(1), m.group(2), int(m.group(3)))] = (float(m.group(4)), float(m.group(5)))
    m2 = re.search(r"LENPPL ckpt=(\S+) deploy=(\S+) L=(\d+) SKIPPED", ln)
    if m2:
        skip.add((m2.group(1), m2.group(2), int(m2.group(3))))

GROUPS = [
    ("Full-attention baselines", [
        ("Llama-3.2-3B-Instruct", "base (no CPT)"), ("a", "A. full causal"),
        ("j_iid_pl", "\\quad$+$sink prefix")]),
    ("Sliding-window baselines", [
        ("b", "B. SWA"), ("h_iid_pl", "\\quad$+$sink prefix"), ("c", "C. SWAA")]),
    ("Ours: symmetric SWA", [
        ("base", "E. S-SWA"), ("token", "\\quad$+$sink token"),
        ("prefix", "\\quad$+$sink prefix"), ("scalar", "\\quad$+$sink scalar")]),
]

def cell(name, dep, L):
    if (name, dep, L) in skip: return "n/a"
    v = data.get((name, dep, L))
    if v is None: return "---"
    p = v[0]
    return "%.0f" % p if p >= 100 else "%.1f" % p

rows = []
ncol = 2 * len(LENS) + 1
for gname, members in GROUPS:
    rows.append("\\multicolumn{%d}{@{}l}{%s}\\\\" % (ncol, gname))
    for name, label in members:
        cells = [cell(name, "full", L) for L in LENS] + [cell(name, "sliding", L) for L in LENS]
        rows.append("%s & %s\\\\" % (label, " & ".join(cells)))
    rows.append("\\midrule")
rows = rows[:-1]

lens_hdr = " & ".join(str(L) for L in LENS)
tex = """\\begin{table*}[!t]
\\centering
\\footnotesize
\\setlength{\\tabcolsep}{5pt}
\\begin{tabular}{l rrrr @{\\hskip 1.4em} rrrr}
\\toprule
 & \\multicolumn{4}{c}{full attention (growing KV)} & \\multicolumn{4}{c}{constant memory (sliding @$1024$)}\\\\
\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}
model (training) & %s & %s\\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\caption{\\textbf{Perplexity vs.\\ evaluation length} (PG-19 test books, first $L$ tokens, $n{\\approx}40$
books; ppl $\\downarrow$), under a growing full-attention KV cache and under the constant-memory sliding
deploy at $W{=}1024$ (registers re-attached for token/prefix rows). The base model collapses under
constant memory once the sequence outgrows the window ($15\\to$$229$ at $16$k): evicting the p$0$ sink
breaks it. Windowed-trained models are length-stable under constant memory, and the sink-prefix rows
stay flat to $16$k; the sink-token register, whose position offsets grow with length, degrades beyond
the trained range --- prefix registers extrapolate where token registers do not. CPT rows are
WikiText-adapted, so absolute PG-19 ppl is domain-shifted for all CPT models; the comparison is the
\\emph{shape} across lengths and the constant-memory gap, not absolute level. n/a $=$ scalar-sink rows
above $4$k (their eager-attention patch exceeds memory).}
\\label{tab:lengthppl}
\\end{table*}
""" % (lens_hdr, lens_hdr, "\n".join(rows))

open(OUT, "w", encoding="utf-8").write(tex)
print("wrote %s (%d measurements, %d skips)" % (OUT, len(data), len(skip)))
