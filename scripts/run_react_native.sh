#!/bin/bash
#SBATCH --job-name=react_native
#SBATCH --partition=nlp
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --time=08:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/reactnative_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/reactnative_%j.log
set -eo pipefail
ROOT=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
source /spack/conda/miniconda3/4.12.0/etc/profile.d/conda.sh
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000)); export WEBSHOP_PORT=$PORT
echo "[native] webshop server on port $PORT ..."
conda run --no-capture-output -p /scratch1/zizhaoh/envs/webshop \
  webshop --host 127.0.0.1 --port $PORT > $ROOT/logs/wsserver_${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /scratch1/zizhaoh/envs/cinw/bin/activate
cd $ROOT
VB=""; [ "${VERBOSE:-0}" = "1" ] && VB="--verbose"
Q=""; [ "${LOAD4BIT:-0}" = "1" ] && Q="--load-4bit"
python -u scripts/react_native_webshop.py --mode "${MODE:?}" --n-episodes "${N_EP:-100}" \
  --sids-file "${SIDS_FILE:?}" --out "${OUT:?}" --model "${MODEL:-Qwen/Qwen3.5-9B}" $Q $VB
echo "DONE react_native $MODE"
