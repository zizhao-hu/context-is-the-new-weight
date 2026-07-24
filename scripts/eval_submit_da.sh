#!/bin/bash
# One self-healing eval pass for the design-A cells (full-history react agents -> REACT=1, like the
# curve_history cells). Skips cells still training / with a live eval / already complete (n>=100).
REPO=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
SIDS=$REPO/../Word2World/data/eval/webshop_test.json
cd "$REPO"
LIVE=$(squeue -u zizhaoh -h -o '%j' 2>/dev/null)
N=${1:-5000}
sub=0
for cell in da_slide_N${N} da_trajrope_N${N} da_recall_N${N}; do
  echo "$LIVE" | grep -qx "$cell" && { echo "skip(training) $cell"; continue; }          # train job name == cell
  [ -f "$REPO/eval_out/$cell/adapter/adapter_config.json" ] || { echo "skip(no-adapter) $cell"; continue; }
  echo "$LIVE" | grep -qx "ev_$cell" && { echo "skip(eval-live) $cell"; continue; }
  out=$REPO/eval_out/rollout_${cell}.json
  done100=$(python3 -c "import json
try:
 d=json.load(open('$out')); print(1 if d.get('$cell',{}).get('n',0)>=100 else 0)
except Exception: print(0)" 2>/dev/null)
  [ "$done100" = "1" ] && { echo "skip(done) $cell"; continue; }
  jid=$(env REACT=1 ADAPTER=$REPO/eval_out/$cell LABEL=$cell SIDS_FILE=$SIDS N_EP=100 OUT=$out \
    sbatch --parsable --partition=nlp --time=06:00:00 --job-name=ev_${cell} scripts/run_rollout.sh 2>/dev/null)
  if [[ "$jid" =~ ^[0-9]+$ ]]; then echo "EVAL $cell -> $jid"; sub=$((sub+1)); else echo "FAIL $cell"; fi
done
echo "da-eval pass: submitted=$sub"
