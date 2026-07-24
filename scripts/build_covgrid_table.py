#!/usr/bin/env python3
"""Build tables/covgrid.tex from the 54-cell coverage grid (covgrid.tsv).

Layout: rows = sink design (6), column groups = unscored rows n in {4,8,16},
columns within group = alpha (S-SWA fraction) in {.10,.50,.75}; cell = suite avg.
Reference column: pure S-SWA (alpha=1) values from the existing cceq rows.
"""
import collections

TSV = "/Users/zizhaohu/.claude/jobs/f25a34dc/tmp/covgrid.tsv"
OUT = "/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/paper/attention-sink/tables/covgrid.tex"

# alpha=1 (pure S-SWA) suite averages from tab:commonsense rows
REF = {"base": 41.0, "tok": 49.6, "pref": 52.4, "scal": 47.2, "rtok": 52.6, "rpref": 48.5}
SINKROW = [("base", "no sink"), ("tok", "$+$sink token"), ("pref", "$+$sink prefix"),
           ("scal", "$+$sink scalar"), ("rtok", "$+$riding token"), ("rpref", "$+$riding prefix")]

data = {}
for ln in open(TSV):
    name, ppl, avg = ln.split("\t")
    parts = name.split("_")            # n4 a10 base
    n = int(parts[0][1:]); al = int(parts[1][1:]); sink = parts[2]
    data[(n, al, sink)] = float(avg)
assert len(data) == 54, len(data)

body = []
for sk, label in SINKROW:
    cells = []
    for n in (4, 8, 16):
        for al in (10, 50, 75):
            cells.append("%.1f" % data[(n, al, sk)])
    cells.append("%.1f" % REF[sk])
    body.append(label + " & " + " & ".join(cells) + "\\\\")

tex = """\\begin{table*}[t]
\\centering
\\footnotesize
\\setlength{\\tabcolsep}{4.6pt}
\\begin{tabular*}{\\textwidth}{@{}l@{\\extracolsep{\\fill}} rrr rrr rrr r@{}}
\\toprule
 & \\multicolumn{3}{c}{$n{=}4$ unscored rows} & \\multicolumn{3}{c}{$n{=}8$} & \\multicolumn{3}{c}{$n{=}16$} & \\multicolumn{1}{c}{S-SWA}\\\\
\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(l){11-11}
sink design & \\multicolumn{1}{c}{$\\alpha{=}.10$} & \\multicolumn{1}{c}{$.50$} & \\multicolumn{1}{c}{$.75$} & \\multicolumn{1}{c}{$.10$} & \\multicolumn{1}{c}{$.50$} & \\multicolumn{1}{c}{$.75$} & \\multicolumn{1}{c}{$.10$} & \\multicolumn{1}{c}{$.50$} & \\multicolumn{1}{c}{$.75$} & \\multicolumn{1}{c}{$\\alpha{=}1$}\\\\
\\midrule
%s
\\bottomrule
\\end{tabular*}
\\caption{Commonsense-suite average across the coverage grid (Llama-3.2-3B CPT, sliding deploy at $W{=}1024$; $\\alpha$ $=$ fraction of symmetric steps, the remaining steps score all but the first $n$ context rows; the $\\alpha{=}1$ column is pure S-SWA from Tab.~\\ref{tab:commonsense}). Any coverage at all recovers most of pure S-SWA's short-context damage: every mixed cell scores $48$--$57$ versus bare S-SWA's $41.0$, with the sink design worth a few points and no strong dependence on $n$ or $\\alpha$ below $1$.}
\\label{tab:covgrid}
\\end{table*}
""" % "\n".join(body)
open(OUT, "w").write(tex)
print("wrote", OUT)
