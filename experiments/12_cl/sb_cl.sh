#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40s
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:30:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/scratch1/zizhaoh/logs/cl_%x_%j.log
set -eo pipefail
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/zizhaoh/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /scratch1/zizhaoh/envs/cinw/bin/activate
cd /scratch1/zizhaoh
python cl_masks.py --mask "$CLMASK" --method "$CLMETH" ${CLEXTRA:-}
