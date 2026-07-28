#!/usr/bin/env python3
"""Build tables/dose.tex: the SWA -> T-SWA transition in one table, sink and task together.

Sweeps C_skip, the number of context rows the loss skips, from 0 (plain SWA) to W (pure T-SWA)
at fixed window, data and scored-token budget (steps scale as C_max/(C_max-C_skip)). Reports the
sink decomposition and the downstream numbers side by side, which is what the toy ablation figure
shows for perplexity only.

Sink numbers: SINKSEM lines (cpt_sinksem.py, sliding deploy, n=12 passages).
Suite: the CPT lm-eval JSONs, same metric rule as the other tables.
"""
import json
import os
import sys

IN = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.claude/jobs/f25a34dc/tmp/cpteval")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["lambada_openai", "piqa", "hellaswag", "winogrande",
         "arc_easy", "arc_challenge", "siqa_pq", "boolq"]

# measured by cpt_sinksem.py: p0 mass and distributed mass, +-SEM over 12 passages
SINK = {
    0:    (0.6896, 0.0277, 0.2566, 0.0527),
    10:   (0.8782, 0.0070, 0.3103, 0.0749),
    256:  (0.6635, 0.0254, 0.3546, 0.0767),
    512:  (0.8140, 0.0105, 0.2435, 0.0458),
    768:  (0.4793, 0.0329, 0.3950, 0.0824),
    1023: (0.9004, 0.0074, 0.4169, 0.0931),
}
# ppl within context (full deploy, deep bin) from the RESULT lines of each run
PPL = {0: 9.21, 10: 9.10, 256: 9.14, 512: 9.09, 768: 9.15, 1023: None}
STEM = {0: "b", 10: "dose10", 256: "dose256", 512: "dose512", 768: "dose768", 1023: "sswa"}
LABEL = {0: "$0$ (SWA)", 1023: "$W$ (T-SWA)"}


def suite(stem):
    p = os.path.join(IN, "%s_sliding.json" % stem)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    v = []
    for t in TASKS:
        if t in d:
            a = d[t].get("acc_norm,none", d[t].get("acc,none"))
            if a is not None:
                v.append(100.0 * a)
    return sum(v) / len(v) if v else None


def main():
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l rr r r}", r"\toprule",
         r"$C_{\mathrm{skip}}$ & p$0$ sink & distributed & ppl & suite\\", r"\midrule"]
    n = 0
    for cs in sorted(SINK):
        p0, p0s, di, dis = SINK[cs]
        s = suite(STEM[cs])
        ppl = PPL[cs]
        L.append("%s & %.2f{\\tiny$\\pm$%.2f} & %.2f{\\tiny$\\pm$%.2f} & %s & %s\\\\" % (
            LABEL.get(cs, "$%d$" % cs), p0, p0s, di, dis,
            ("%.2f" % ppl) if ppl is not None else "---",
            ("%.1f" % s) if s is not None else "---"))
        n += 1
    L += [r"\bottomrule", r"\end{tabular}",
          r"""\caption{The SWA to T-SWA transition in one sweep (Llama-3.2-3B CPT, $W{=}1024$,
$C_{\max}{=}2048$). $C_{\mathrm{skip}}$ is how many context rows the loss skips; steps scale as
$C_{\max}/(C_{\max}-C_{\mathrm{skip}})$ so every row trains on the same number of scored tokens.
Sink columns are attention mass under the sliding deploy ($\pm1$ SEM over $12$ passages); ppl is
the within-context deep bin; suite is the mean of the eight zero-shot tasks. Perplexity is flat
across the whole transition while the suite degrades, so the loss rule moves where attention parks
without changing language quality. The sink columns do not vary monotonically, and with one run per
point the sweep does not resolve the shape of that dependence.}""",
          r"\label{tab:dose}", r"\end{table}", ""]
    open(os.path.join(ROOT, "paper/attention-sink/tables/dose.tex"), "w").write("\n".join(L))
    print("wrote tables/dose.tex (%d rows)" % n)


if __name__ == "__main__":
    main()
