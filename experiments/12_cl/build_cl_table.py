#!/usr/bin/env python3
"""Harvest CLEVAL lines from the cl_masks.py logs and emit the paper table.

Usage: build_cl_table.py <log dir or glob> > tables/cl.tex

Per config (mask, method) the log carries, for stages 0..4 and tasks
{wikitext, gsm8k, tofu, arc, fineweb}: CLEVAL stage=k task=name ppl=x.

Table columns per row: final ppl on the four tasks (stage 4), their mean,
forgetting = mean over tasks of (ppl at stage 4 minus ppl just after the task's own
stage), and the fineweb retention probe at stage 4. The base row is stage 0.
"""
import glob
import re
import sys

TASKS = ["wikitext", "gsm8k", "tofu", "arc"]
STAGE_OF = {"wikitext": 1, "gsm8k": 2, "tofu": 3, "arc": 4}
METHODS = [("naive", None), ("replay", r"\quad$+$replay (ER)"), ("ewc", r"\quad$+$EWC"),
           ("lwf", r"\quad$+$LwF"), ("l2", r"\quad$+$L2-SP")]
ROWS = [(m, meth, (label if meth != "naive" else head))
        for m, head in [("a", "A. full causal"), ("b", "B. SWA"), ("t", "E. T-SWA")]
        for meth, label in METHODS]

def parse(path):
    ev = {}
    done = False
    for ln in open(path, errors="ignore"):
        m = re.match(r"CLEVAL stage=(\d+) task=(\S+) ppl=([\d.]+)", ln)
        if m:
            ev[(int(m.group(1)), m.group(2))] = float(m.group(3))
        if ln.startswith("CLDONE"):
            done = True
    return ev, done

def main():
    pats = sys.argv[1:] or ["logs"]
    logs = []
    for p in pats:
        logs += glob.glob(p + "/cl_cl_*_*.log") if not any(c in p for c in "*?") else glob.glob(p)
    runs = {}
    base = None
    for path in sorted(logs):
        m = re.search(r"cl_cl_([abt])_(naive|replay|l2)_\d+\.log$", path)
        if not m:
            continue
        ev, done = parse(path)
        if not done:
            print("%% skipping incomplete %s" % path, file=sys.stderr)
            continue
        runs[(m.group(1), m.group(2))] = ev
        if base is None and all((0, t) in ev for t in TASKS + ["fineweb"]):
            base = ev
    out = []
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering")
    out.append(r"\scriptsize")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l rrrr r r r}")
    out.append(r"\toprule")
    out.append(r" & \multicolumn{4}{c}{final ppl after the full sequence ($\downarrow$)} & & & \\")
    out.append(r"\cmidrule(lr){2-5}")
    out.append(r"training & \multicolumn{1}{c}{wikitext} & \multicolumn{1}{c}{gsm8k} & "
               r"\multicolumn{1}{c}{tofu} & \multicolumn{1}{c}{arc} & \multicolumn{1}{c}{avg} & "
               r"\multicolumn{1}{c}{forget} & \multicolumn{1}{c}{fineweb}\\")
    out.append(r"\midrule")
    if base:
        cells = ["%.2f" % base[(0, t)] for t in TASKS]
        avg = sum(base[(0, t)] for t in TASKS) / 4
        out.append("base (no CPT) & %s & %.2f & --- & %.2f\\\\" %
                   (" & ".join(cells), avg, base[(0, "fineweb")]))
        out.append(r"\midrule")
    for mask, meth, label in ROWS:
        ev = runs.get((mask, meth))
        if not ev:
            out.append("%s & \\multicolumn{7}{c}{---}\\\\" % label)
            continue
        fin = [ev[(4, t)] for t in TASKS]
        forget = sum(ev[(4, t)] - ev[(STAGE_OF[t], t)] for t in TASKS) / 4
        out.append("%s & %s & %.2f & %+.2f & %.2f\\\\" %
                   (label, " & ".join("%.2f" % v for v in fin), sum(fin) / 4,
                    forget, ev[(4, "fineweb")]))
    out.append(r"\bottomrule")
    out.append(r"\end{tabular*}")
    out.append(r"\caption{Sequential continued pretraining over four tasks (wikitext, gsm8k, tofu,")
    out.append(r"arc): final held-out perplexity per task, their mean, forgetting (mean rise from")
    out.append(r"just after each task's own stage to the end of the sequence), and perplexity on")
    out.append(r"the never-trained fineweb probe. Each design evaluates under its matched deploy;")
    out.append(r"the base row is under full attention.}")
    out.append(r"\label{tab:cl}")
    out.append(r"\end{table*}")
    print("\n".join(out))

if __name__ == "__main__":
    main()
