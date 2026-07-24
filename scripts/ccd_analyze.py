"""Analyze CCD lifetime runs: hill-climbing (reward vs episode) + first/last-third vs baselines."""
import json, os, sys
R = "/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/eval_out"
MODES = ["hybrid", "reward_gated", "everything", "frozen"]
# static baselines (Score = mean reward*100 already /100 here; use 0-1)
BASE = {"Act (prompt)": 0.69, "ReAct (prompt)": 0.12, "full-hist ET (frozen)": 0.61}

def load(m):
    f = f"{R}/ccd_{m}.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)); log = d.get("log", [])
    rs = [r["reward"] for r in log]
    return d, rs

print(f"{'mode':<16}{'n':>5}{'overall':>9}{'first3':>8}{'last3':>8}{'SR':>7}{'climb':>8}")
for m in MODES:
    out = load(m)
    if not out: print(f"{m:<16}  (no data yet)"); continue
    d, rs = out
    if not rs: print(f"{m:<16}  (empty)"); continue
    n = len(rs); t = max(1, n//3)
    f3 = sum(rs[:t])/t; l3 = sum(rs[-t:])/t; ov = sum(rs)/n
    sr = sum(1 for r in rs if r > 0.95)/n
    print(f"{m:<16}{n:>5}{ov:>9.3f}{f3:>8.3f}{l3:>8.3f}{sr:>7.2f}{l3-f3:>+8.3f}")
print("\nstatic baselines (Score, 0-1):")
for k, v in BASE.items(): print(f"  {k:<24}{v:.3f}")
# verdict
h = load("hybrid")
if h and h[1]:
    rs = h[1]; t = max(1, len(rs)//3); l3 = sum(rs[-t:])/len(rs[-t:]); ov = sum(rs)/len(rs)
    best_base = max(BASE.values())
    print(f"\nCCD-hybrid last-third={l3:.3f} overall={ov:.3f} | best baseline={best_base:.3f} | "
          f"BEATS_ALL={'YES' if l3 > best_base else 'no (last3 vs '+str(best_base)+')'}")
