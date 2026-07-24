#!/bin/bash
#SBATCH --job-name=react_alf
#SBATCH --partition=nlp
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/reactalf_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/reactalf_%j.log
# ReAct-paper ALFWorld replication (Table 3), frozen Qwen3.5-9B in the alfworld venv.
set -eo pipefail
ROOT=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ALFWORLD_DATA=/scratch1/zizhaoh/alfworld_data
source /scratch1/zizhaoh/envs/alfworld/bin/activate
cd $ROOT
VB=""; [ "${VERBOSE:-0}" = "1" ] && VB="--verbose"
python -u scripts/react_baseline_alfworld.py \
  --config baselines/ReAct/base_config.yaml \
  --prompts baselines/ReAct/prompts/alfworld_3prompts.json \
  --n-games "${N_GAMES:-134}" --out "${OUT:?}" $VB
echo "DONE react_alfworld"
