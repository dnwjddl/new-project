#!/bin/bash
#SBATCH --job-name=vft-smoke
#SBATCH --partition=gigabyte_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.out
#SBATCH --error=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.err

set -euo pipefail

PROJECT=/home/dnwjddl/project1_temporal_audit
PYTHON=/home/dnwjddl/miniconda3/envs/videoflextok/bin/python
ANNOTATION=/dataset/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_SAMPLES=${MAX_SAMPLES:-4}
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun "$PYTHON" "$PROJECT/extract_ek100_videoflextok.py" \
  --annotation "$ANNOTATION" \
  --split train \
  --conditions ordered \
  --output-dir "$PROJECT/features/videoflextok_smoke" \
  --shard-id 0 \
  --num-shards 8 \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --overwrite
