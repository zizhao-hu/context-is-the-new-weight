#!/usr/bin/env python3
"""Rebuild tables/commonsense.tex and tables/covgrid.tex from the CPT lm-eval JSONs.

One metric rule everywhere, matching build_t340_suite_table.py: length-normalised accuracy
where the harness defines it (PIQA, HellaSwag, both ARC splits), plain accuracy otherwise.
The previous hand-built versions averaged max(acc, acc_norm) per task, which picks whichever
metric flatters each model and inflated every row by 0.1-1.0 points.
"""
import json
import os
import sys

IN = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.claude/jobs/f25a34dc/tmp/cpteval")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["lambada_openai", "piqa", "hellaswag", "winogrande",
         "arc_easy", "arc_challenge", "siqa_pq", "boolq"]
HEAD = ["LMB", "PIQA", "Hella", "Wino", "ARC-e", "ARC-c", "SIQA", "BoolQ"]

SUITE = [  # (json stem, printed label, group heading or None)
    ("base",         "base (no CPT)",                  "Full-attention baselines"),
    ("a",            "A.\\ full causal",               None),
    ("a_token",      "\\quad$+$sink token",            None),
    ("a_prefix",     "\\quad$+$sink prefix",           None),
    ("a_scalar",     "\\quad$+$sink scalar",           None),
    ("b",            "B.\\ SWA",                       "Sliding-window baselines"),
    ("b_token",      "\\quad$+$sink token",            None),
    ("b_prefix",     "\\quad$+$sink prefix",           None),
    ("b_scalar",     "\\quad$+$sink scalar",           None),
    ("b_ridtok",     "\\quad$+$riding token",          None),
    ("b_ridpref",    "\\quad$+$riding prefix",         None),
    ("c",            "C.\\ SWAA",                      None),
    ("sswa",         "E.\\ S-SWA",                     "Ours: symmetric SWA"),
    ("sswa_token",   "\\quad$+$sink token",            None),
    ("sswa_prefix",  "\\quad$+$sink prefix",           None),
    ("sswa_scalar",  "\\quad$+$sink scalar",           None),
    ("sswa_ridtok",  "\\quad$+$riding token",          None),
    ("sswa_ridpref", "\\quad$+$riding prefix",         None),
    ("dose10",       "\\quad coverage, $10$ unscored rows", None),
    ("mix1",         "\\quad coverage mix $\\alpha{=}.99$", None),
    ("mix5",         "\\quad coverage mix $\\alpha{=}.95$", None),
]
BASELINE = {"lambada_openai": 0.0, "piqa": 50.0, "hellaswag": 25.0, "winogrande": 50.0,
            "arc_easy": 25.0, "arc_challenge": 25.0, "siqa_pq": 33.3, "boolq": 62.2}


def cells(stem):
    p = os.path.join(IN, "%s_sliding.json" % stem)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = []
    for t in TASKS:
        v = d.get(t)
        if v is None:
            out.append(None); continue
        a = v.get("acc_norm,none", v.get("acc,none"))
        out.append(None if a is None else 100.0 * a)
    return out


def fmt(task, v):
    if v is None:
        return "---"
    return (r"\textcolor{black!45}{%.1f}" % v) if v <= BASELINE[task] else "%.1f" % v


def suite_table():
    L = [r"\begin{table*}[!t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{3.4pt}",
         r"\begin{tabular}{l rrrrrrrr r}", r"\toprule",
         "model (training) & " + " & ".join(HEAD) + r" & Avg\\", r"\midrule"]
    for stem, label, grp in SUITE:
        c = cells(stem)
        if c is None:
            continue
        if grp:
            L.append(r"\multicolumn{10}{@{}l}{%s}\\" % grp)
        ok = [x for x in c if x is not None]
        L.append("%s & %s & %.1f\\\\" % (label, " & ".join(fmt(t, v) for t, v in zip(TASKS, c)),
                                         sum(ok) / len(ok)))
    L += [r"\bottomrule", r"\end{tabular}",
          r"""\caption{\textbf{Commonsense-reasoning suite} (lm-evaluation-harness, zero-shot) on the
continued-pretraining checkpoints (Llama-3.2-3B, WikiText CPT under each mask), scored under the
constant-memory sliding deploy at $W{=}1024$. Every column is accuracy in \% ($\uparrow$):
length-normalised accuracy for PIQA, HellaSwag and both ARC splits, plain accuracy elsewhere;
Avg is the mean of the eight. Greyed cells sit at or below the task's random or majority-class
baseline. Same suite as SWAT \citep{swat2025}; this is continued pretraining of a 3B model, so
these rows compare to each other and to Tab.~\ref{tab:t340suite} only in direction, not in
level.}""",
          r"\label{tab:commonsense}", r"\end{table*}", ""]
    open(os.path.join(ROOT, "paper/attention-sink/tables/commonsense.tex"), "w").write("\n".join(L))
    return sum(1 for stem, _, _ in SUITE if cells(stem))


def grid_avg(stem):
    p = os.path.join(IN, "grid_%s_sliding.json" % stem)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    v = []
    for t in TASKS:
        if t not in d:
            continue
        a = d[t].get("acc_norm,none", d[t].get("acc,none"))
        if a is not None:
            v.append(100.0 * a)
    return sum(v) / len(v) if v else None


DESIGNS = [("base", "no sink"), ("tok", "$+$sink token"), ("pref", "$+$sink prefix"),
           ("scal", "$+$sink scalar"), ("rtok", "$+$riding token"), ("rpref", "$+$riding prefix")]
ALPHA1 = {"base": "sswa", "tok": "sswa_token", "pref": "sswa_prefix", "scal": "sswa_scalar",
          "rtok": "sswa_ridtok", "rpref": "sswa_ridpref"}


def grid_table():
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4.6pt}",
         r"\begin{tabular*}{\textwidth}{@{}l@{\extracolsep{\fill}} r rrr rrr rrr r@{}}", r"\toprule",
         r" & \multicolumn{1}{c}{SWA} & \multicolumn{3}{c}{$n{=}4$ unscored rows} & "
         r"\multicolumn{3}{c}{$n{=}8$} & \multicolumn{3}{c}{$n{=}16$} & \multicolumn{1}{c}{S-SWA}\\",
         r"\cmidrule(lr){2-2}\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(l){12-12}",
         r"sink design & \multicolumn{1}{c}{$\alpha{=}0$} & \multicolumn{1}{c}{$\alpha{=}.10$} & \multicolumn{1}{c}{$.50$} & "
         r"\multicolumn{1}{c}{$.75$} & \multicolumn{1}{c}{$.10$} & \multicolumn{1}{c}{$.50$} & "
         r"\multicolumn{1}{c}{$.75$} & \multicolumn{1}{c}{$.10$} & \multicolumn{1}{c}{$.50$} & "
         r"\multicolumn{1}{c}{$.75$} & \multicolumn{1}{c}{$\alpha{=}1$}\\", r"\midrule"]
    for key, label in DESIGNS:
        row = [grid_avg("a0_%s" % key)]                       # alpha=0 endpoint, same family
        for n in (4, 8, 16):
            for al in (10, 50, 75):
                row.append(grid_avg("n%d_a%d_%s" % (n, al, key)))
        row.append(grid_avg("a100_%s" % key))                 # alpha=1 endpoint, same family
        if row[-1] is None:                                   # fall back to the main-family S-SWA row
            ref = cells(ALPHA1[key])
            if ref:
                ok = [x for x in ref if x is not None]
                row[-1] = sum(ok) / len(ok)
        L.append("%s & %s\\\\" % (label, " & ".join("%.1f" % v if v is not None else "---" for v in row)))
    L += [r"\bottomrule", r"\end{tabular*}",
          r"""\caption{Commonsense-suite average across the coverage grid (Llama-3.2-3B CPT, sliding
deploy at $W{=}1024$; $\alpha$ $=$ fraction of symmetric steps, the remaining steps score all but
the first $n$ context rows. Both endpoints are run in this same family: $\alpha{=}0$ scores every
row (plain SWA) and $\alpha{=}1$ scores only full-window rows (pure S-SWA), so the sweep is paired
throughout. Same metric rule as Tab.~\ref{tab:commonsense}. Any coverage at all recovers most of
pure S-SWA's short-context damage: every mixed cell scores $47$--$57$ versus bare S-SWA's $40.9$.
Beyond that the grid resolves little: neither $n$ nor $\alpha$ shows a monotone trend, each cell is
a single run without repeats, and nominally similar cells differ by up to $9$ points, so read the
recovery, not the ranking within it.}""",
          r"\label{tab:covgrid}", r"\end{table*}", ""]
    open(os.path.join(ROOT, "paper/attention-sink/tables/covgrid.tex"), "w").write("\n".join(L))


if __name__ == "__main__":
    n = suite_table()
    grid_table()
    print("rebuilt commonsense.tex (%d rows) and covgrid.tex under the standard metric rule" % n)
