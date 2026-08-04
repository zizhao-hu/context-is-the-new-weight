#!/bin/bash
#SBATCH --partition=nlp
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/sink4eval_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/sink4eval_%j.log
set -eo pipefail
ROOT=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
EXP=$ROOT/experiments/05_sink_4way
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/zizhaoh/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /scratch1/zizhaoh/envs/cinw/bin/activate
cd $ROOT
python $EXP/measure_sink1.py --model "$1" $2 --label "$3" --out "$4"
echo "EVAL DONE $3"
