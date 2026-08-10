#!/bin/bash
#SBATCH --job-name=ego4d-nlq-clip
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
PYTHON=/home/dnwjddl/miniconda3/envs/ar_video/bin/python
SPLIT=${1:?Pass train or val}
ANNOTATION=/dataset/Ego4D/v2/annotations/nlq_${SPLIT}.json
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT/features/ego4d_nlq_clip_4s}
MAX_CLIPS=${MAX_CLIPS:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
STRIDE_SECONDS=${STRIDE_SECONDS:-4}
DECODE_THREADS=${DECODE_THREADS:-1}
DECODE_WORKERS=${DECODE_WORKERS:-4}
mkdir -p "$PROJECT/logs" "$OUTPUT_DIR"

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun "$PYTHON" "$PROJECT/extract_ego4d_nlq_clip.py" \
  --annotation "$ANNOTATION" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --stride-seconds "$STRIDE_SECONDS" \
  --decode-threads "$DECODE_THREADS" \
  --decode-workers "$DECODE_WORKERS" \
  --max-clips "$MAX_CLIPS" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --overwrite
