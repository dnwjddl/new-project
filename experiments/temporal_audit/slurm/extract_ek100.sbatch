#!/bin/bash
#SBATCH --job-name=ek100-viclip
#SBATCH --partition=gigabyte_a6000,suma_rtx4090,dell_rtx3090,base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --exclude=node04,cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/dnwjddl/project1_temporal_audit/logs/%x-%A_%a.out
#SBATCH --error=/home/dnwjddl/project1_temporal_audit/logs/%x-%A_%a.err

set -euo pipefail

PROJECT=/home/dnwjddl/project1_temporal_audit
PYTHON=/home/dnwjddl/miniconda3/envs/ar_video/bin/python
SPLIT=${1:?Pass train or validation}
ANNOTATION_ROOT=/dataset/EK100/epic-kitchens-100-annotations
mkdir -p "$PROJECT/logs" "$PROJECT/features/viclip_l14_8f"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

if [[ "$SPLIT" == "train" ]]; then
  ANNOTATION="$ANNOTATION_ROOT/EPIC_100_train.csv"
  CONDITIONS=ordered
else
  ANNOTATION="$ANNOTATION_ROOT/EPIC_100_validation.csv"
  CONDITIONS=ordered,reverse,shuffle,single,sparse,freeze
fi

srun "$PYTHON" "$PROJECT/extract_ek100_viclip.py" \
  --annotation "$ANNOTATION" \
  --split "$SPLIT" \
  --conditions "$CONDITIONS" \
  --output-dir "$PROJECT/features/viclip_l14_8f" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --overwrite
