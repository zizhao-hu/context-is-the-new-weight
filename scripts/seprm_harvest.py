#!/usr/bin/env python3
"""Harvest the separator-removal study (iterative token removal, masks a+b).

Pulls per-round artifacts from the cluster logs into one TSV:
  round, mask, dropped_ids, top1_id, top1_repr, top1_mass, p0, dist_total,
  wikitext ppl (full deep / stream deep), hotpot F1/Acc (full + own deploy).
Refuses to emit rows for rounds whose logs are incomplete.
Run: python scripts/seprm_harvest.py   (ssh endeavour required)
"""
import subprocess, re, sys

OUT = "/Users/zizhaohu/.claude/jobs/f25a34dc/tmp/autoresearch_seprm/seprm_results.tsv"

def sh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "endeavour", cmd],
                          capture_output=True, text=True, timeout=120).stdout

rows = []
for mask in ("a", "b"):
    for it in range(0, 6):
        topk = sh("cat /scratch1/zizhaoh/seprm_topk_%s_it%d.txt 2>/dev/null" % (mask, it))
        if "SINKTOPK_DONE" not in topk:
            continue
        m = re.search(r"SINKTOPK \S+ .* \| p0 ([\d.]+) \| dist_total ([\d.]+)", topk)
        t = re.search(r"SINKTOK \S+ id=(\d+) repr=(.+?) mass=([\d.]+)", topk)
        row = {"mask": mask, "iter": it, "p0": m.group(1), "dist": m.group(2),
               "top1_id": t.group(1) if t else "", "top1_repr": t.group(2) if t else "",
               "top1_mass": t.group(3) if t else ""}
        if it > 0:
            log = sh("grep -hE 'HOTPOT model|RESULT cpt|SEPRM iter' "
                     "$(ls -t /scratch1/zizhaoh/logs/seprm_seprm_%s_it%d_*.log 2>/dev/null | head -1) 2>/dev/null" % (mask, it))
            hp = re.search(r"deploy=full W=\d+ \| F1 ([\d.]+) \| EM [\d.]+ \| Acc ([\d.]+)", log)
            hp2 = re.search(r"deploy=(?:windowed|streaming) W=\d+ \| F1 ([\d.]+) \| EM [\d.]+ \| Acc ([\d.]+)", log)
            rc = re.search(r"full\(avg [\d.]+ deep ([\d.]+)\)", log)
            dr = re.search(r"SEPRM iter=\d+ mask=\w+ drop=\[([^\]]*)\]", log)
            row.update({"f1_full": hp.group(1) if hp else "", "acc_full": hp.group(2) if hp else "",
                        "f1_dep": hp2.group(1) if hp2 else "", "acc_dep": hp2.group(2) if hp2 else "",
                        "ppl_full": rc.group(1) if rc else "", "dropped": dr.group(1) if dr else ""})
        rows.append(row)

cols = ["mask", "iter", "dropped", "top1_id", "top1_repr", "top1_mass", "p0", "dist",
        "ppl_full", "f1_full", "acc_full", "f1_dep", "acc_dep"]
with open(OUT, "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
print("wrote %s (%d rows)" % (OUT, len(rows)))
for r in rows:
    print(r)
