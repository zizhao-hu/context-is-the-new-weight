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
           ("lwf", r"\quad$+$LwF")]   # L2-SP runs exist but the row is inert; stated in text
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

def metrics(ev):
    """final avg ppl, forgetting (best-reference, first 3 tasks), BWT (signed, first 3), fineweb"""
    fin = [ev[(4, t)] for t in TASKS]
    first3 = TASKS[:3]
    F = sum(ev[(4, t)] - min(ev[(s, t)] for s in range(STAGE_OF[t], 5)) for t in first3) / 3
    B = sum(ev[(STAGE_OF[t], t)] - ev[(4, t)] for t in first3) / 3
    return sum(fin) / 4, F, B, ev[(4, "fineweb")], fin

def main():
    pats = sys.argv[1:] or ["logs"]
    logs = []
    for p in pats:
        logs += glob.glob(p + "/cl_cl_*.log") if not any(c in p for c in "*?") else glob.glob(p)
    runs = {}
    base = None
    for path in sorted(logs):
        m = re.search(r"cl_cl_([abt])_(naive|replay|l2|ewc|lwf)_\d+\.log$", path)
        if not m:
            continue
        ev, done = parse(path)
        if not done:
            print("%% skipping incomplete %s" % path, file=sys.stderr)
            continue
        runs[(m.group(1), m.group(2))] = ev
        if base is None and all((0, t) in ev for t in TASKS + ["fineweb"]):
            base = ev
    hyb = {}
    for path in sorted(logs):
        m = re.search(r"cl_cl_hyb_([abt])(?:_(naive|replay|ewc|lwf))?_\d+\.log$", path)
        if not m:
            continue
        ev, done = parse(path)
        if done:
            hyb[(m.group(1), m.group(2) or "naive")] = ev

    # ---- main table: metric columns ----
    out = []
    out.append(r"\begin{table}[tp]")
    out.append(r"\centering")
    out.append(r"\scriptsize")
    out.append(r"\setlength{\tabcolsep}{3pt}")
    out.append(r"\renewcommand{\arraystretch}{0.90}")
    out.append(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r r r r}")
    out.append(r"\toprule")
    out.append(r"training & \multicolumn{1}{c}{ppl} & \multicolumn{1}{c}{forget} & "
               r"\multicolumn{1}{c}{BWT} & \multicolumn{1}{c}{fineweb}\\")
    out.append(r"\midrule")
    def base_rows(dst, full_ev, slide_ev):
        if full_ev:
            avg = sum(full_ev[(0, t)] for t in TASKS) / 4
            dst.append("base (no CPT) & %.2f & --- & --- & %.2f\\\\" % (avg, full_ev[(0, "fineweb")]))
        if slide_ev and all((0, t) in slide_ev for t in TASKS + ["fineweb"]):
            avg = sum(slide_ev[(0, t)] for t in TASKS) / 4
            dst.append("\\quad sliding deploy & %.2f & --- & --- & %.2f\\\\" %
                       (avg, slide_ev[(0, "fineweb")]))
    if base:
        base_rows(out, base, runs.get(("b", "naive")))
        out.append(r"\midrule")
    for mask, meth, label in ROWS:
        ev = runs.get((mask, meth))
        if not ev:
            out.append("%s & \\multicolumn{4}{c}{---}\\\\" % label)
            continue
        avg, F, B, fw, _ = metrics(ev)
        out.append("%s & %.2f & %.2f & %+.2f & %.2f\\\\" % (label, avg, F, B, fw))
    if hyb:
        out.append(r"\midrule")
        out.append(r"\multicolumn{5}{@{}l}{Qwen3.5-9B hybrid (softmax layers trained)}\\")
        hb = hyb.get(("a", "naive")) or next(iter(hyb.values()))
        base_rows(out, hb, hyb.get(("b", "naive")))
        for mask, head in [("a", "A. full causal"), ("b", "B. SWA"), ("t", "E. T-SWA")]:
            for meth, label in METHODS:
                if meth == "l2":
                    continue
                ev = hyb.get((mask, meth))
                if not ev:
                    continue
                avg, F, B, fw, _ = metrics(ev)
                out.append("%s & %.2f & %.2f & %+.2f & %.2f\\\\" %
                           (head if meth == "naive" else label, avg, F, B, fw))
    out.append(r"\bottomrule")
    out.append(r"\end{tabular*}")
    out.append(r"\caption{Continual learning over the four-task sequence: final perplexity averaged")
    out.append(r"over tasks, forgetting (rise from each task's best post-learning perplexity),")
    out.append(r"backward transfer (positive helps), and the never-trained fineweb probe, each")
    out.append(r"design under its matched deploy. Base rows give the unadapted model under the")
    out.append(r"full-attention and sliding deploys; the hybrid trains $50$ steps per stage.")
    out.append(r"Per-task numbers in")
    out.append(r"App.~\ref{app:cltasks}.}")
    out.append(r"\label{tab:cl}")
    out.append(r"\end{table}")
    print("\n".join(out))

    # ---- appendix table: per-task matrix ----
    ap = []
    ap.append(r"\begin{table*}[t]")
    ap.append(r"\centering")
    ap.append(r"\scriptsize")
    ap.append(r"\setlength{\tabcolsep}{4pt}")
    ap.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l rrrr r}")
    ap.append(r"\toprule")
    ap.append(r"training & \multicolumn{1}{c}{wikitext} & \multicolumn{1}{c}{gsm8k} & "
              r"\multicolumn{1}{c}{tofu} & \multicolumn{1}{c}{arc} & \multicolumn{1}{c}{fineweb}\\")
    ap.append(r"\midrule")
    def base_rows_tasks(dst, full_ev, slide_ev):
        if full_ev:
            dst.append("base (no CPT) & %s & %.2f\\\\" %
                       (" & ".join("%.2f" % full_ev[(0, t)] for t in TASKS), full_ev[(0, "fineweb")]))
        if slide_ev and all((0, t) in slide_ev for t in TASKS + ["fineweb"]):
            dst.append("\\quad sliding deploy & %s & %.2f\\\\" %
                       (" & ".join("%.2f" % slide_ev[(0, t)] for t in TASKS), slide_ev[(0, "fineweb")]))
    if base:
        base_rows_tasks(ap, base, runs.get(("b", "naive")))
        ap.append(r"\midrule")
    for mask, meth, label in ROWS:
        ev = runs.get((mask, meth))
        if not ev:
            continue
        ap.append("%s & %s & %.2f\\\\" %
                  (label, " & ".join("%.2f" % ev[(4, t)] for t in TASKS), ev[(4, "fineweb")]))
    if hyb:
        ap.append(r"\midrule")
        ap.append(r"\multicolumn{6}{@{}l}{Qwen3.5-9B hybrid (softmax layers trained)}\\")
        base_rows_tasks(ap, hyb.get(("a", "naive")), hyb.get(("b", "naive")))
        for mask, head in [("a", "A. full causal"), ("b", "B. SWA"), ("t", "E. T-SWA")]:
            for meth, label in METHODS:
                if meth == "l2":
                    continue
                ev = hyb.get((mask, meth))
                if not ev:
                    continue
                ap.append("%s & %s & %.2f\\\\" %
                          (head if meth == "naive" else label,
                           " & ".join("%.2f" % ev[(4, t)] for t in TASKS), ev[(4, "fineweb")]))
    ap.append(r"\bottomrule")
    ap.append(r"\end{tabular*}")
    ap.append(r"\caption{Per-task final perplexity after the full continual sequence, for every")
    ap.append(r"training and method of Tab.~\ref{tab:cl}.}")
    ap.append(r"\label{tab:cltasks}")
    ap.append(r"\end{table*}")
    import io, os
    with open(os.environ.get("CL_APPENDIX_OUT", "cl_tasks.tex"), "w") as f:
        f.write("\n".join(ap) + "\n")

if __name__ == "__main__":
    main()
