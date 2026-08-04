#!/bin/bash
#SBATCH --partition=nlp
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/all8_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/all8_%j.log
set -eo pipefail
ROOT=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
EXP=$ROOT/experiments/09_litmask
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/zizhaoh/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /scratch1/zizhaoh/envs/cinw/bin/activate
cd $ROOT
MASK=$1
BND=$2
EXTRA="${@:3}"
echo "ALL8_START mask=$MASK boundary=$BND extra=$EXTRA"
python $EXP/toy_all8.py --mask $MASK --boundary $BND $EXTRA
echo "ALL8_JOBDONE mask=$MASK boundary=$BND"
