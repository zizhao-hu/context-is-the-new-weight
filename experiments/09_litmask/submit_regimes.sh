#!/bin/bash
# Submit our regimes e/f/g (sliding_history, warmup, persistent) through the same harness for Table 1.
cd /project2/jessetho_1732/zizhaoh/context-is-the-new-weight
R=experiments/09_litmask/run_regimes.sh
echo "=== CPT regimes (Qwen2.5-0.5B) ==="
for SC in sliding_history warmup persistent; do
  for W in 128 256 512; do sbatch $R cptreg $SC $W --steps 600 | sed "s/$/  [cptreg $SC W$W]/"; done
done
echo "=== TOY regimes (4-layer GPT) ==="
for SC in sliding_history warmup persistent; do
  for W in 32 64 128; do sbatch $R toyreg $SC $W --steps 3000 | sed "s/$/  [toyreg $SC W$W]/"; done
done
echo "=== queued ==="; squeue -u zizhaoh -h | wc -l
