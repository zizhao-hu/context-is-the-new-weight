#!/bin/bash
# Submit the full Fig-2 b/c/d matrix: {full, windowed(b), swaa(c), longformer(d)} x window-sweep,
# for both CPT (Qwen2.5-0.5B) and TOY (from-scratch GPT). Each job evals full/sliding/StreamingLLM.
cd /project2/jessetho_1732/zizhaoh/context-is-the-new-weight
R=experiments/09_litmask/run_litmask.sh
echo "=== CPT (Qwen2.5-0.5B, ctx 2048) ==="
for M in full windowed swaa longformer; do
  for W in 128 256 512; do
    sbatch $R cpt $M $W --steps 600 | sed "s/$/  [cpt $M W$W]/"
  done
done
echo "=== TOY (4-layer GPT, ctx 256) ==="
for M in full windowed swaa longformer; do
  for W in 32 64 128; do
    sbatch $R toy $M $W --steps 3000 | sed "s/$/  [toy $M W$W]/"
  done
done
echo "=== queued ==="; squeue -u zizhaoh -h | wc -l
