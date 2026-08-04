#!/bin/bash
#SBATCH --partition=nlp
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/sink_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/sink_%j.log
set -eo pipefail
ROOT=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
EXP=$ROOT/experiments/04_attention_sink_cause
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /scratch1/zizhaoh/envs/cinw/bin/activate
cd $ROOT
# A: base (causal pretraining only)
python $EXP/measure_sink.py --label base --out $EXP/sink_base.json
# B: experience/instruction-tuned adapter (causal, amplified by tuning)
python $EXP/measure_sink.py --adapter $ROOT/eval_out/curve_history_N3000 --label experience --out $EXP/sink_experience.json
echo "DONE attention-sink A/B"
