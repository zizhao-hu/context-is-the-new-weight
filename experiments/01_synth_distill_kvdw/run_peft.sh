#!/bin/bash
#SBATCH --job-name=cinw-peft
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/%x_%j.log

# Train + eval one PEFT method (lora|prompt|prefix) on one context.
# Required env: CONTEXT, METHOD.
# Phases: train (phase2x) → ROUGE val (phase3x) → 3-way activations (phase4x)
#         → MMLU-Pro (phase5x).

set -eo pipefail
module purge
module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface
export HF_HUB_ENABLE_XET=0
export TOKENIZERS_PARALLELISM=false

source /scratch1/zizhaoh/envs/cinw/bin/activate
cd /project2/jessetho_1732/zizhaoh/context-is-the-new-weight

CONTEXT="${CONTEXT:?CONTEXT env var required}"
METHOD="${METHOD:?METHOD env var required (lora|prompt|prefix)}"
CONFIG="experiments/01_synth_distill_kvdw/config.yaml"

SAVES_ROOT=/scratch1/zizhaoh/context-is-the-new-weight/saves
DATA_ROOT=/scratch1/zizhaoh/context-is-the-new-weight/data/synth
OUT_ROOT=/scratch1/zizhaoh/context-is-the-new-weight/outputs/01_synth_distill_kvdw

TMPCFG=$(mktemp --suffix=.yaml)
python - <<EOF >"$TMPCFG"
import yaml, sys
cfg = yaml.safe_load(open("$CONFIG"))
cfg["data_root"]  = "$DATA_ROOT"
cfg["saves_root"] = "$SAVES_ROOT"
cfg["out_root"]   = "$OUT_ROOT"
sys.stdout.write(yaml.safe_dump(cfg))
EOF

echo "[run_peft.sh] CONTEXT=$CONTEXT  METHOD=$METHOD  CONFIG=$TMPCFG"

echo "===== PHASE 2x: train $METHOD adapter ====="
python experiments/01_synth_distill_kvdw/phase2x_peft.py \
  --config "$TMPCFG" --context "$CONTEXT" --method "$METHOD"

echo "===== PHASE 3x: ROUGE validation ====="
python experiments/01_synth_distill_kvdw/phase3x_rouge_peft.py \
  --config "$TMPCFG" --context "$CONTEXT" --method "$METHOD"

echo "===== PHASE 4x: 3-way activations ====="
python experiments/01_synth_distill_kvdw/phase4x_peft_activations.py \
  --config "$TMPCFG" --context "$CONTEXT" --method "$METHOD"

echo "===== PHASE 5x: MMLU-Pro ====="
python experiments/01_synth_distill_kvdw/phase5x_mmlu_peft.py \
  --config "$TMPCFG" --context "$CONTEXT" --method "$METHOD"

rm -f "$TMPCFG"
echo "[run_peft.sh] DONE  context=$CONTEXT  method=$METHOD"
