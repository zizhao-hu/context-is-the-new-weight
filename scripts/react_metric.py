import json, os
R = "/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/eval_out"
NS = [150,300,500,1000,2000,3000,5000,8000,10000,15000,20000]
def stats(f):
    d = json.load(open(f)); v = list(d.values())[0]
    eps = v.get("episodes", []); n = len(eps)
    if n == 0: return None
    rew = [e["reward"] for e in eps]
    score = 100*sum(rew)/n
    sr_exact = 100*sum(1 for r in rew if r >= 1.0)/n   # ReAct SR == reward 1.0
    sr_95 = 100*sum(1 for r in rew if r > 0.95)/n
    return score, sr_exact, sr_95, n
hdr = "cell".ljust(22) + "Score".rjust(7) + "SR(=1)".rjust(8) + "SR(>.95)".rjust(9) + "n".rjust(5)
print(hdr)
for t in ["history", "obsact", "obsreact"]:
    print("-- " + t)
    for nN in NS:
        f = R + "/rollout_curve_" + t + "_N" + str(nN) + ".json"
        if os.path.exists(f):
            s = stats(f)
            if s:
                name = ("curve_" + t + "_N" + str(nN)).ljust(22)
                print(name + ("%.1f"%s[0]).rjust(7) + ("%.1f"%s[1]).rjust(8) + ("%.1f"%s[2]).rjust(9) + str(s[3]).rjust(5))
