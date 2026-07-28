#!/usr/bin/env python3
"""Build tables/t340suite.tex from the 340M from-scratch lm-eval JSONs.

SWAT-comparable setting: 341M Mistral-style LM, 15B FineWeb tokens, C=2048, W=1024,
each mask pretrained from scratch. One row per variant, one column per suite task.

Input: <indir>/t340_<tag>_<deploy>.json written by lmeval_sswa.py (keys are task names,
values carry acc,none / acc_norm,none / perplexity,none).
Rows are emitted only for variants whose JSON exists, so the table can be built while
the remaining runs are still training.
"""
import json
import os
import sys

IN = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.claude/jobs/f25a34dc/tmp/t340eval")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper/attention-sink/tables/t340suite.tex")

# (json tag, deploy, printed label, group)
ROWS = [
    ("full",  "full",    "A. full causal",                    "Full-attention pretraining"),
    ("swa",   "sliding", "B. SWA",                            "Sliding-window pretraining"),
    ("sswa",  "sliding", "E. T-SWA",                          "Ours: truncated SWA"),
    ("mix10", "sliding", r"\quad coverage mix $\alpha{=}.10$", "Ours: truncated SWA"),
    ("mix50", "sliding", r"\quad coverage mix $\alpha{=}.50$", "Ours: truncated SWA"),
    ("mix75", "sliding", r"\quad coverage mix $\alpha{=}.75$", "Ours: truncated SWA"),
    ("mix0_n8", "sliding", r"\quad coverage only, $n{=}8$ unscored", "Ours: truncated SWA"),
    ("mix0_n256", "sliding", r"\quad coverage only, $n{=}256$ unscored", "Ours: truncated SWA"),
]
TASKS = [("piqa", "PIQA"), ("hellaswag", "Hella"), ("winogrande", "Wino"),
         ("arc_easy", "ARC-e"), ("arc_challenge", "ARC-c"), ("siqa_pq", "SIQA"),
         ("boolq", "BoolQ")]
# random / majority-class baseline per task; cells at or below it carry no signal
BASE = {"lambada_openai": 0.0, "piqa": 50.0, "hellaswag": 25.0, "winogrande": 50.0,
        "arc_easy": 25.0, "arc_challenge": 25.0, "siqa_pq": 33.3, "boolq": 62.2}


def mark(task, v):
    if v is None:
        return "---"
    return (r"\textcolor{black!45}{%.1f}" % v) if v <= BASE.get(task, 0.0) else "%.1f" % v


def load(tag, deploy):
    p = os.path.join(IN, "t340_%s_%s.json" % (tag, deploy))
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def acc(d, task):
    v = d.get(task)
    if v is None:
        return None
    a = v.get("acc_norm,none", v.get("acc,none"))
    return None if a is None else 100.0 * a


def main():
    lines = [r"\begin{table*}[!t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3.4pt}",
             r"\begin{tabular}{l r rrrrrrr r}", r"\toprule",
             "model (pretraining) & LMB & " +
             " & ".join(t[1] for t in TASKS) + r" & Avg\\", r"\midrule"]
    group = None
    n = 0
    for tag, deploy, label, grp in ROWS:
        d = load(tag, deploy)
        if d is None:
            continue
        if grp != group:
            group = grp
            lines.append(r"\multicolumn{10}{@{}l}{%s}\\" % grp)
        lmb = d.get("lambada_openai", {})
        lacc = lmb.get("acc,none")
        cells = [acc(d, t[0]) for t in TASKS]
        got = [c for c in cells if c is not None]
        allacc = got + ([100.0 * lacc] if lacc is not None else [])
        avg = sum(allacc) / len(allacc) if allacc else float("nan")
        row = "%s & %s & %s & %.1f" % (
            label,
            ("%.1f" % (100.0 * lacc)) if lacc is not None else "---",
            " & ".join(mark(t[0], c) for t, c in zip(TASKS, cells)),
            avg)
        lines.append(row + r"\\")
        n += 1
    lines += [r"\bottomrule", r"\end{tabular}",
              r"""\caption{\textbf{Commonsense-reasoning suite} (lm-evaluation-harness, zero-shot) on models
pretrained \emph{from scratch} under each mask: 341M parameters, 15B FineWeb tokens, Llama-2
vocabulary, global batch $0.5$M tokens, train context $C{=}2048$, window $W{=}1024$, identical
data order and budget across rows. Task columns are scored under each model's own deploy (full
attention for A, constant-memory sliding at $W{=}1024$ for the rest), but no item is long
enough to reach that window, so the deploy is inert here and the columns compare weights. Every column is accuracy in \% ($\uparrow$): length-normalised accuracy for PIQA, HellaSwag
and both ARC splits, plain accuracy for LAMBADA, WinoGrande, SIQA and BoolQ. Avg $=$ mean of the
eight. Greyed cells sit at or below the task's random or majority-class baseline, where the
ordering between rows carries no signal. We follow SWAT's protocol \citep{swat2025} in model size, token budget, batch and
vocabulary, but train at $C{=}2048$ rather than $4096$ and on FineWeb rather than their corpus,
so these rows are comparable to each other and not to their published numbers. All rows are
sink-free: they isolate the loss rule.}""",
              r"\label{tab:t340suite}", r"\end{table*}", ""]
    open(OUT, "w").write("\n".join(lines))
    print("wrote %s (%d data rows)" % (OUT, n))


if __name__ == "__main__":
    main()
