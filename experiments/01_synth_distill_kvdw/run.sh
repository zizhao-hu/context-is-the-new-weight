#!/bin/bash
#SBATCH --job-name=cinw
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/%x_%j.log

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

# Override out paths to point to scratch (fast, not backed up).
export CONTEXT="${CONTEXT:?CONTEXT env var required (e.g. haiku, pirate, concise, fewshot_translate_fr, factual)}"
CONFIG="experiments/01_synth_distill_kvdw/config.yaml"

# Use scratch-backed output paths so checkpoints land on fast storage.
SAVES_ROOT=/scratch1/zizhaoh/context-is-the-new-weight/saves
DATA_ROOT=/scratch1/zizhaoh/context-is-the-new-weight/data/synth
OUT_ROOT=/scratch1/zizhaoh/context-is-the-new-weight/outputs/01_synth_distill_kvdw

# Patch the config in-place at runtime via a tiny temp config that points to
# scratch storage. (Cleaner than env-var threading through 4 scripts.)
TMPCFG=$(mktemp --suffix=.yaml)
python - <<EOF >"$TMPCFG"
import yaml, sys
cfg = yaml.safe_load(open("$CONFIG"))
cfg["data_root"]  = "$DATA_ROOT"
cfg["saves_root"] = "$SAVES_ROOT"
cfg["out_root"]   = "$OUT_ROOT"
sys.stdout.write(yaml.safe_dump(cfg))
EOF

echo "[run.sh] CONTEXT=$CONTEXT  CONFIG=$TMPCFG"

echo "===== PHASE 1 ====="
python experiments/01_synth_distill_kvdw/phase1_dataset.py --config "$TMPCFG" --context "$CONTEXT"

echo "===== PHASE 2 ====="
python experiments/01_synth_distill_kvdw/phase2_distill.py --config "$TMPCFG" --context "$CONTEXT"

echo "===== PHASE 3 ====="
python experiments/01_synth_distill_kvdw/phase3_verify.py --config "$TMPCFG" --context "$CONTEXT"

echo "===== PHASE 4 ====="
python experiments/01_synth_distill_kvdw/phase4_kvdw.py --config "$TMPCFG" --context "$CONTEXT"

rm -f "$TMPCFG"
echo "[run.sh] DONE  context=$CONTEXT"
