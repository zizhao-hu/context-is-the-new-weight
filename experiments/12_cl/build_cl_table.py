#!/usr/bin/env python3
"""Harvest CLEVAL lines from the cl_masks.py logs and emit the paper table.

Usage: build_cl_table.py <log dir or glob> > tables/cl.tex

Per config (mask, method) the log carries, for stages 0..4 and tasks
{wikitext, gsm8k, tofu, arc, fineweb}: CLEVAL stage=k task=name ppl=x.

Both tables group method-major so the three masks sit adjacent: the claim is
about masks, and the naive block must read 1.86 / 4.60 / 1.11 in consecutive
rows. Bold marks the unique best mask within the naive group on the
deploy-invariant metrics (forgetting, BWT); ties bold nothing.
"""
import glob
import re
import sys

TASKS = ["wikitext", "gsm8k", "tofu", "arc"]
STAGE_OF = {"wikitext": 1, "gsm8k": 2, "tofu": 3, "arc": 4}
MASKS = [("a", "A. full causal"), ("b", "B. SWA"), ("t", "E. T-SWA"),
         ("bs", "B $+$ sink"), ("ts", "E $+$ sink")]
GROUPS = [("naive", r"\textit{naive}"), ("replay", r"\textit{$+$replay (ER)}"),
          ("ewc", r"\textit{$+$EWC}"), ("lwf", r"\textit{$+$LwF}")]
# L2-SP runs exist but the row is inert; stated in text

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

def task_forget(ev, t):
    """rise of task t's ppl from its best post-learning value to the end"""
    return ev[(4, t)] - min(ev[(s, t)] for s in range(STAGE_OF[t], 5))

def unique_best(scores, pick):
    """mask key of the strictly best score, or None on a tie"""
    if not scores:
        return None
    tgt = pick(scores.values())
    win = [m for m, v in scores.items() if v == tgt]
    return win[0] if len(win) == 1 else None

def base_rows(dst, full_ev, slide_ev, fmt):
    if full_ev:
        dst.append(fmt("base (no CPT)", full_ev))
    if slide_ev and all((0, t) in slide_ev for t in TASKS + ["fineweb"]):
        dst.append(fmt(r"\quad sliding deploy", slide_ev))

def metric_fmt(label, ev):
    avg = sum(ev[(0, t)] for t in TASKS) / 4
    return "%s & %.2f & --- & --- & %.2f\\\\" % (label, avg, ev[(0, "fineweb")])

def task_fmt(label, ev):
    cells = []
    for t in TASKS[:3]:
        cells += ["%.2f" % ev[(0, t)], "---"]
    cells += ["%.2f" % ev[(0, "arc")], "%.2f" % ev[(0, "fineweb")]]
    return "%s & %s\\\\" % (label, " & ".join(cells))

def metric_block(dst, get):
    """method-major rows: an italic method line, then the three masks"""
    vals = {(m, meth): metrics(get(m, meth))
            for m, _ in MASKS for meth, _ in GROUPS if get(m, meth)}
    nF = {m: vals[(m, "naive")][1] for m, _ in MASKS if (m, "naive") in vals}
    nB = {m: vals[(m, "naive")][2] for m, _ in MASKS if (m, "naive") in vals}
    bF, bB = unique_best(nF, min), unique_best(nB, max)
    for meth, glabel in GROUPS:
        if not any((m, meth) in vals for m, _ in MASKS):
            continue
        dst.append(r"\multicolumn{5}{@{}l}{%s}\\" % glabel)
        for m, mhead in MASKS:
            if (m, meth) not in vals:
                continue
            avg, F, B, fw, _ = vals[(m, meth)]
            fs, bs = "%.2f" % F, "%+.2f" % B
            if meth == "naive" and m == bF:
                fs = r"\textbf{%s}" % fs
            if meth == "naive" and m == bB:
                bs = r"\textbf{%s}" % bs
            dst.append(r"\quad %s & %.2f & %s & %s & %.2f\\" % (mhead, avg, fs, bs, fw))

def task_block(dst, get):
    for meth, glabel in GROUPS:
        rows = [(m, mhead, get(m, meth)) for m, mhead in MASKS if get(m, meth)]
        if not rows:
            continue
        dst.append(r"\multicolumn{9}{@{}l}{%s}\\" % glabel)
        for m, mhead, ev in rows:
            cells = []
            for t in TASKS[:3]:
                cells += ["%.2f" % ev[(4, t)], "%+.2f" % task_forget(ev, t)]
            cells += ["%.2f" % ev[(4, "arc")], "%.2f" % ev[(4, "fineweb")]]
            dst.append(r"\quad %s & %s\\" % (mhead, " & ".join(cells)))

def main():
    pats = sys.argv[1:] or ["logs"]
    logs = []
    for p in pats:
        logs += glob.glob(p + "/cl_cl_*.log") if not any(c in p for c in "*?") else glob.glob(p)
    runs = {}
    base = None
    for path in sorted(logs):
        m = re.search(r"cl_cl_(a|b|t|bs|ts)_(naive|replay|l2|ewc|lwf)_(\d+)\.log$", path)
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

    # ---- main table: side-by-side metric blocks (softmax | hybrid) ----
    svals = {k: metrics(ev) for k, ev in runs.items()}
    hvals = {k: metrics(ev) for k, ev in hyb.items()}

    def bolds(vals):
        # compare at display precision so a tie at 0.03 never bolds one of the pair
        nF = {m: round(vals[(m, "naive")][1], 2) for m, _ in MASKS if (m, "naive") in vals}
        nB = {m: round(vals[(m, "naive")][2], 2) for m, _ in MASKS if (m, "naive") in vals}
        return unique_best(nF, min), unique_best(nB, max)
    sbF, sbB = bolds(svals)
    hbF, hbB = bolds(hvals)

    def cells_metric(v, bF, bB):
        if v is None:
            return ["---"] * 4
        avg, F, B, fw, _ = v
        fs, bs = "%.2f" % F, "%+.2f" % B
        if bF:
            fs = r"\textbf{%s}" % fs
        if bB:
            bs = r"\textbf{%s}" % bs
        return ["%.2f" % avg, fs, bs, "%.2f" % fw]

    def cells_base(ev):
        if not (ev and all((0, t) in ev for t in TASKS + ["fineweb"])):
            return ["---"] * 4
        return ["%.2f" % (sum(ev[(0, t)] for t in TASKS) / 4), "---", "---",
                "%.2f" % ev[(0, "fineweb")]]

    out = []
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering")
    out.append(r"\scriptsize")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\renewcommand{\arraystretch}{0.87}")
    out.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l rrrr rrrr}")
    out.append(r"\toprule")
    out.append(r" & \multicolumn{4}{c}{Qwen2.5-0.5B softmax} & "
               r"\multicolumn{4}{c}{Qwen3.5-9B hybrid}\\")
    out.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    hdr = (r"\multicolumn{1}{c}{ppl$\downarrow$} & \multicolumn{1}{c}{forget$\downarrow$} & "
           r"\multicolumn{1}{c}{BWT$\uparrow$} & \multicolumn{1}{c}{fineweb$\downarrow$}")
    out.append("training & %s & %s\\\\" % (hdr, hdr))
    out.append(r"\midrule")
    row = lambda label, left, right: out.append(
        "%s & %s\\\\" % (label, " & ".join(left + right)))
    row("base (no CPT)", cells_base(base), cells_base(hyb.get(("a", "naive"))))
    row(r"\quad sliding deploy", cells_base(runs.get(("b", "naive"))),
        cells_base(hyb.get(("b", "naive"))))
    out.append(r"\midrule")
    for meth, glabel in GROUPS:
        out.append(r"\multicolumn{9}{@{}l}{%s}\\" % glabel)
        for m, mhead in MASKS:
            row(r"\quad %s" % mhead,
                cells_metric(svals.get((m, meth)),
                             meth == "naive" and m == sbF, meth == "naive" and m == sbB),
                cells_metric(hvals.get((m, meth)),
                             meth == "naive" and m == hbF, meth == "naive" and m == hbB))
    out.append(r"\bottomrule")
    out.append(r"\end{tabular*}")
    out.append(r"\caption{Continual learning over the four-task sequence: final perplexity averaged")
    out.append(r"over tasks, forgetting (rise from each task's best post-learning perplexity),")
    out.append(r"backward transfer (positive helps), and the never-trained fineweb probe; matched")
    out.append(r"deploy, equal supervised-token budget, hybrid $50$ steps per stage. Base rows: the")
    out.append(r"unadapted model under each deploy. Per-task numbers in App.~\ref{app:cltasks};")
    out.append(r"bold, best naive mask.}")
    out.append(r"\label{tab:cl}")
    out.append(r"\end{table*}")
    print("\n".join(out))

    # ---- appendix table: per-task matrix ----
    ap = []
    ap.append(r"\begin{table*}[t]")
    ap.append(r"\centering")
    ap.append(r"\scriptsize")
    ap.append(r"\setlength{\tabcolsep}{4pt}")
    ap.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l rr rr rr r r}")
    ap.append(r"\toprule")
    ap.append(r" & \multicolumn{2}{c}{wikitext} & \multicolumn{2}{c}{gsm8k} & "
              r"\multicolumn{2}{c}{tofu} & \multicolumn{1}{c}{arc} & \multicolumn{1}{c}{fineweb}\\")
    ap.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    ap.append(r"training & \multicolumn{1}{c}{ppl$\downarrow$} & \multicolumn{1}{c}{F$\downarrow$} & "
              r"\multicolumn{1}{c}{ppl$\downarrow$} & \multicolumn{1}{c}{F$\downarrow$} & "
              r"\multicolumn{1}{c}{ppl$\downarrow$} & \multicolumn{1}{c}{F$\downarrow$} & "
              r"\multicolumn{1}{c}{ppl$\downarrow$} & \multicolumn{1}{c}{ppl$\downarrow$}\\")
    ap.append(r"\midrule")
    if base:
        base_rows(ap, base, runs.get(("b", "naive")), task_fmt)
        ap.append(r"\midrule")
    task_block(ap, lambda m, meth: runs.get((m, meth)))
    if hyb:
        ap.append(r"\midrule")
        ap.append(r"\multicolumn{9}{@{}l}{Qwen3.5-9B hybrid (softmax layers trained)}\\")
        base_rows(ap, hyb.get(("a", "naive")), hyb.get(("b", "naive")), task_fmt)
        task_block(ap, lambda m, meth: hyb.get((m, meth)))
    ap.append(r"\bottomrule")
    ap.append(r"\end{tabular*}")
    ap.append(r"\caption{Per-task breakdown after the full continual sequence, for every training")
    ap.append(r"and method of Tab.~\ref{tab:cl}: final perplexity and, for each previous task, its")
    ap.append(r"forgetting F (rise from the task's best post-learning perplexity; ARC is learned")
    ap.append(r"last so it cannot be forgotten).}")
    ap.append(r"\label{tab:cltasks}")
    ap.append(r"\end{table*}")
    import os
    with open(os.environ.get("CL_APPENDIX_OUT", "cl_tasks.tex"), "w") as f:
        f.write("\n".join(ap) + "\n")

if __name__ == "__main__":
    main()
