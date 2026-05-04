#!/bin/bash
# Submit one SLURM job per (context, method) for the PEFT comparison experiments.
# Run from the repo root on Endeavour:
#   bash experiments/01_synth_distill_kvdw/submit_peft.sh
# Override CTXS or METHODS via env, e.g.:
#   METHODS="lora" bash experiments/01_synth_distill_kvdw/submit_peft.sh
set -e
CTXS=(${CTXS:-haiku pirate concise fewshot_translate_fr factual})
METHODS=(${METHODS:-lora prompt})
for m in "${METHODS[@]}"; do
  for c in "${CTXS[@]}"; do
    jid=$(sbatch --parsable --export=CONTEXT="$c",METHOD="$m" experiments/01_synth_distill_kvdw/run_peft.sh)
    echo "[$m/$c] submitted job $jid"
  done
done
echo "---"
echo "Watch with: squeue -u zizhaoh -o '%.10i %.30j %.8T %.10M %R'"
