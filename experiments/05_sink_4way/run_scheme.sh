#!/bin/bash
#SBATCH --partition=nlp
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/sink4_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight/logs/sink4_%j.log
set -eo pipefail
ROOT=/project2/jessetho_1732/zizhaoh/context-is-the-new-weight
EXP=$ROOT/experiments/05_sink_4way
module purge && module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/zizhaoh/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /scratch1/zizhaoh/envs/cinw/bin/activate
cd $ROOT
SCHEME=$1
DATA=${2:-wikitext}
TAG=$SCHEME; [ "$DATA" != "wikitext" ] && TAG=${SCHEME}_${DATA}
DA=""; [ "$DATA" != "wikitext" ] && DA="--data $DATA"
if [ "$SCHEME" = "base" ]; then
  python $EXP/measure_sink1.py --model Qwen/Qwen2.5-0.5B $DA --label $TAG --out $EXP/sink/$TAG.json
else
  ST=""; [ "$SCHEME" = "startup" ] && ST="--startup"
  [ "$SCHEME" = "sliding_history" ] && ST="--history 255"     # eval with a real history prefix (matches training)
  python $EXP/train_4way.py --scheme $SCHEME --data $DATA --out $EXP/models/$TAG --steps 800 --warmup_steps 400
  python $EXP/measure_sink1.py --model $EXP/models/$TAG $ST $DA --label $TAG --out $EXP/sink/$TAG.json
fi
echo "DONE $TAG"
