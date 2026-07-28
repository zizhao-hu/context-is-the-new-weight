#!/usr/bin/env python3
"""Build tables/t340covgrid.tex: coverage grid at from-scratch pretraining scale.

Axes actually run at 341M/15B scale:
  alpha (fraction of T-SWA steps) in {0, .10, .50, .75, 1} at n=8 unscored rows
  n (unscored rows) in {4, 8, 16} at alpha=.50
alpha=0 is the SWA run, alpha=1 is the T-SWA run, so the sweep is anchored at both ends.

Reads the same lm-eval JSONs as build_t340_suite_table.py and reports the suite average
(mean over the accuracy tasks), matching the CPT coverage grid's metric.
"""
import json
import os
import sys

IN = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.claude/jobs/f25a34dc/tmp/t340eval")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/tables/t340covgrid.tex")

TASKS = ["piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "siqa_pq", "boolq"]
ALPHA_ROW = [("swa", "sliding", r"$\alpha{=}0$ (SWA)"),
             ("mix10", "sliding", r"$.10$"),
             ("mix50", "sliding", r"$.50$"),
             ("mix75", "sliding", r"$.75$"),
             ("sswa", "sliding", r"$1$ (T-SWA)")]
N_ROW = [("mix50_n4", "sliding", r"$n{=}4$"),
         ("mix50", "sliding", r"$n{=}8$"),
         ("mix50_n16", "sliding", r"$n{=}16$")]


def avg(tag, deploy):
    p = os.path.join(IN, "t340_%s_%s.json" % (tag, deploy))
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    vals = []
    for t in TASKS:
        v = d.get(t)
        if v is None:
            continue
        a = v.get("acc_norm,none", v.get("acc,none"))
        if a is not None:
            vals.append(100.0 * a)
    lmb = d.get("lambada_openai", {}).get("acc,none")
    if lmb is not None:
        vals.append(100.0 * lmb)
    return sum(vals) / len(vals) if vals else None


def cells(rows):
    out = []
    for tag, deploy, label in rows:
        a = avg(tag, deploy)
        out.append((label, "%.1f" % a if a is not None else "---"))
    return out


def main():
    al = cells(ALPHA_ROW)
    nr = cells(N_ROW)
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{l %s}" % ("r" * len(al)), r"\toprule",
         r"\multicolumn{%d}{@{}l}{Symmetric fraction $\alpha$ (at $n{=}8$ unscored rows)}\\" % (len(al) + 1),
         " & ".join([""] + [c[0] for c in al]) + r"\\",
         "suite avg & " + " & ".join(c[1] for c in al) + r"\\",
         r"\midrule",
         r"\multicolumn{%d}{@{}l}{Unscored rows $n$ (at $\alpha{=}.50$)}\\" % (len(al) + 1),
         " & ".join([""] + [c[0] for c in nr] + [""] * (len(al) - len(nr))) + r"\\",
         "suite avg & " + " & ".join([c[1] for c in nr] + ["" for _ in range(len(al) - len(nr))]) + r"\\",
         r"\bottomrule", r"\end{tabular}",
         r"""\caption{Coverage grid at from-scratch pretraining scale (341M, 15B FineWeb tokens,
$C{=}2048$, $W{=}1024$, sliding deploy). $\alpha$ is the fraction of T-SWA steps; the
remaining steps leave the first $n$ context rows unscored. Endpoints are the SWA and T-SWA runs
of Tab.~\ref{tab:t340suite}. Suite avg $=$ mean accuracy over the tasks of that table.}""",
         r"\label{tab:t340covgrid}", r"\end{table}", ""]
    open(OUT, "w").write("\n".join(L))
    got = sum(1 for _, v in al + nr if v != "---")
    print("wrote %s (%d/%d cells populated)" % (OUT, got, len(al) + len(nr)))


if __name__ == "__main__":
    main()
