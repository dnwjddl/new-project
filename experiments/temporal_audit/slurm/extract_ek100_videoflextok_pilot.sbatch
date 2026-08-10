#!/bin/bash
#SBATCH --job-name=ek100-vft-pilot
#SBATCH --partition=gigabyte_a6000,suma_rtx4090,dell_rtx3090,base_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --exclude=node04,cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/dnwjddl/project1_temporal_audit/logs/%x-%A_%a.out
#SBATCH --error=/home/dnwjddl/project1_temporal_audit/logs/%x-%A_%a.err

set -euo pipefail

PROJECT=/home/dnwjddl/project1_temporal_audit
PYTHON=/home/dnwjddl/miniconda3/envs/videoflextok/bin/python
SPLIT=${1:?Pass train or validation}
ANNOTATION_ROOT=/dataset/EK100/epic-kitchens-100-annotations
OUTPUT_DIR=$PROJECT/features/videoflextok_k600_pilot16k
mkdir -p "$PROJECT/logs" "$OUTPUT_DIR"

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

if [[ "$SPLIT" == "train" ]]; then
  ANNOTATION="$ANNOTATION_ROOT/EPIC_100_train.csv"
  CONDITIONS=ordered
  MAX_TOTAL=16000
else
  ANNOTATION="$ANNOTATION_ROOT/EPIC_100_validation.csv"
  CONDITIONS=ordered,reverse
  MAX_TOTAL=0
fi

srun "$PYTHON" "$PROJECT/extract_ek100_videoflextok.py" \
  --annotation "$ANNOTATION" \
  --split "$SPLIT" \
  --conditions "$CONDITIONS" \
  --output-dir "$OUTPUT_DIR" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --batch-size 12 \
  --max-total-samples "$MAX_TOTAL" \
  --overwrite
