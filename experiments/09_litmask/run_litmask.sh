#!/bin/bash
#SBATCH --partition=nlp
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/litmask_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/litmask_%j.log
set -eo pipefail
ROOT=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
EXP=$ROOT/experiments/09_litmask
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/zizhaoh/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /scratch1/zizhaoh/envs/cinw/bin/activate
cd $ROOT
KIND=$1            # cpt | toy
MASK=$2           # full | windowed | swaa | longformer
W=$3              # window size
EXTRA="${@:4}"
echo "LITMASK_START kind=$KIND mask=$MASK W=$W extra=$EXTRA"
if [ "$KIND" = "cpt" ]; then
  python $EXP/cpt_masks.py --train_mask $MASK --window $W $EXTRA
else
  python $EXP/toy_masks.py --train_mask $MASK --window $W $EXTRA
fi
echo "LITMASK_DONE kind=$KIND mask=$MASK W=$W"
