#!/bin/bash
# Design-A experiment: intra-trajectory persistent RoPE on full-history (sliding-window) training.
# 3 new cells at N (+ reuse curve_history_N${N} = truncated baseline already trained):
#   da_slide   : --win 4096 --stride 2048              (full traj, sliding window, STANDARD rope per window)
#   da_trajrope: --win 4096 --stride 2048 --traj-rope  (design A: intra-traj absolute positions, reset/traj)
#   da_recall  : --win 4096 --stride 2048 --traj-rope --recall
# iid SFT (STREAM=0), same early-stop convergence criterion + shared val set as the scaling curve.
REPO=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
D=/scratch1/zizhaoh/context-is-the-new-weight/data
VAL=$D/val_think_300.json
ES="--val-file $VAL --patience 4 --eval-every 15 --min-steps 15 --max-steps 400"
WIN="--win 4096 --stride 2048"
cd "$REPO"
N=${1:-5000}
data=curve_react_N${N}
declare -A CELLS=(
  [da_slide_N${N}]="$WIN"
  [da_trajrope_N${N}]="$WIN --traj-rope"
  [da_recall_N${N}]="$WIN --traj-rope --recall"
)
for cell in "${!CELLS[@]}"; do
  rm -rf "$REPO/eval_out/$cell" "$REPO/eval_out/rollout_${cell}.json" 2>/dev/null
  jid=$(AUG_FILE=$D/${data}.json STREAM=0 EPOCHS=20 EXTRA_FLAGS="${CELLS[$cell]} $ES" OUT=$REPO/eval_out/$cell \
    sbatch --parsable --partition=nlp_hiprio --time=08:00:00 --job-name=${cell} scripts/run_train_react.sh 2>/dev/null)
  if [[ "$jid" =~ ^[0-9]+$ ]]; then echo "TRAIN $cell -> $jid"; else echo "FAIL $cell"; fi
done
