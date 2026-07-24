#!/bin/bash
#SBATCH --job-name=ws_rollout
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/rollout_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/rollout_%j.log
set -eo pipefail
ROOT=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
source /spack/conda/miniconda3/4.12.0/etc/profile.d/conda.sh
# unique port per job so co-located jobs don't collide on the env-server socket
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000)); export WEBSHOP_PORT=$PORT
# 1) WebShop env server (CPU) in background, its own conda env
echo "[rollout] starting webshop server on port $PORT (loads ~1.18M product index, can take minutes)..."
conda run --no-capture-output -p /scratch1/zizhaoh/envs/webshop \
  webshop --host 127.0.0.1 --port $PORT > $ROOT/logs/wsserver_${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT
# 2) model rollout (GPU) in cinw, talks to server over HTTP
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /scratch1/zizhaoh/envs/cinw/bin/activate
cd $ROOT
EXTRA=""
[ -n "$ADAPTER" ] && EXTRA="--adapter $ADAPTER --label ${LABEL:-adapter}"
[ "${REACT:-0}" = "1" ] && EXTRA="$EXTRA --react"
[ -n "$SIDS_FILE" ] && EXTRA="$EXTRA --sids-file $SIDS_FILE"
[ -n "$THINK_MODE" ] && EXTRA="$EXTRA --think-mode $THINK_MODE"
[ "${VERBOSE:-0}" = "1" ] && EXTRA="$EXTRA --verbose"
[ "${NO_HIST:-0}" = "1" ] && EXTRA="$EXTRA --no-history"
[ "${LOAD4BIT:-0}" = "1" ] && EXTRA="$EXTRA --load-4bit"
python scripts/rollout_webshop.py --n-episodes "${N_EP:-3}" --cells "${CELLS:-all}" --obs0-label "${OBS0_LABEL:-obs_0}" $EXTRA \
  --model "${MODEL:-Qwen/Qwen3.5-9B}" --session-id-start "${SID0:-500}" --out "${OUT:-/scratch1/zizhaoh/context-is-the-new-weight/outputs/rollout_webshop.json}"
echo "DONE rollout"
