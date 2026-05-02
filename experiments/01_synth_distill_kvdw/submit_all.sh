#!/bin/bash
# Submit one SLURM job per context for the sprint experiments.
# Run from the repo root on Endeavour:
#   bash experiments/01_synth_distill_kvdw/submit_all.sh
set -e
CTXS=(haiku pirate concise fewshot_translate_fr factual)
for c in "${CTXS[@]}"; do
  jid=$(sbatch --parsable --export=CONTEXT="$c" experiments/01_synth_distill_kvdw/run.sh)
  echo "[$c] submitted job $jid"
done
echo "---"
echo "Watch with: squeue -u zizhaoh -o '%.10i %.30j %.8T %.10M %R'"
