#!/bin/bash
#SBATCH --job-name=ek100-vft
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/dnwjddl/project1_temporal_audit/logs/%x-%A_%a.out
#SBATCH --error=/home/dnwjddl/project1_temporal_audit/logs/%x-%A_%a.err

set -euo pipefail

PROJECT=/home/dnwjddl/project1_temporal_audit
PYTHON=/home/dnwjddl/miniconda3/envs/videoflextok/bin/python
SPLIT=${1:?Pass train or validation}
ANNOTATION_ROOT=/dataset/EK100/epic-kitchens-100-annotations
mkdir -p "$PROJECT/logs" "$PROJECT/features/videoflextok_k600_17f"

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

if [[ "$SPLIT" == "train" ]]; then
  ANNOTATION="$ANNOTATION_ROOT/EPIC_100_train.csv"
  CONDITIONS=ordered
else
  ANNOTATION="$ANNOTATION_ROOT/EPIC_100_validation.csv"
  CONDITIONS=ordered,reverse,shuffle,single
fi

srun "$PYTHON" "$PROJECT/extract_ek100_videoflextok.py" \
  --annotation "$ANNOTATION" \
  --split "$SPLIT" \
  --conditions "$CONDITIONS" \
  --output-dir "$PROJECT/features/videoflextok_k600_17f" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --overwrite
