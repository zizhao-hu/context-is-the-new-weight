#!/bin/bash
# Re-invoke operator when all 3 design-A eval results are complete (n>=100), or after the cap.
MAXPOLL=${1:-30}
REPO=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
for i in $(seq 1 "$MAXPOLL"); do
  done=$(ssh -o ConnectTimeout=20 endeavour "python3 - <<'PY'
import json,os
R='$REPO/eval_out'; d=0
for c in ['da_slide_N5000','da_trajrope_N5000','da_recall_N5000']:
    f=f'{R}/rollout_{c}.json'
    try:
        if json.load(open(f)).get(c,{}).get('n',0)>=100: d+=1
    except Exception: pass
print(d)
PY" 2>/dev/null)
  echo "[wait poll $i] da_evals_done=${done:-?}/3"
  [ "${done:-0}" -ge 3 ] && { echo "[wait] design-A evals complete"; exit 0; }
  sleep 300
done
echo "[wait] maxpoll reached"
