#!/bin/bash
# Re-invoke when EITHER both prompted 27B baselines reach n=100, OR the ET-27B training stops.
MAXPOLL=${1:-50}
REPO=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
for i in $(seq 1 "$MAXPOLL"); do
  read pb tr <<<"$(ssh -o ConnectTimeout=20 endeavour "
    p=\$(python3 - <<'PY'
import json
R='$REPO/eval_out'; n=0
for m in ['react','act']:
    try:
        d=json.load(open(f'{R}/rollout_reactbase_{m}_27b.json'));
        if list(d.values())[0].get('n',0)>=100: n+=1
    except Exception: pass
print(n)
PY
)
    t=\$(squeue -u zizhaoh -h -o '%j' 2>/dev/null | grep -c '^et27b')
    echo \${p:-0} \${t:-0}" 2>/dev/null)"
  echo "[wait poll $i] prompted27b_done=$pb/2  et27b_training=$tr"
  [ "${pb:-0}" -ge 2 ] && { echo "[wait] prompted 27B baselines complete"; exit 0; }
  [ "${tr:-1}" -eq 0 ] && { echo "[wait] ET-27B training stopped"; exit 0; }
  sleep 300
done
echo "[wait] maxpoll reached"
