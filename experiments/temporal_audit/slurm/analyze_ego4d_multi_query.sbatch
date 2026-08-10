#!/bin/bash
#SBATCH --job-name=ego4d-mq-analysis
#SBATCH --partition=gigabyte_a6000,suma_rtx4090,dell_rtx3090,base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --exclude=node04,cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.out
#SBATCH --error=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.err

set -euo pipefail

PROJECT=/home/dnwjddl/project1_temporal_audit
PYTHON=/home/dnwjddl/miniconda3/envs/ar_video/bin/python
FEATURE_DIR=${FEATURE_DIR:-$PROJECT/features/ego4d_nlq_clip_4s}
TEXT_CACHE=${TEXT_CACHE:-$PROJECT/features/ego4d_nlq_clip_text.npz}
OUTPUT=${OUTPUT:-$PROJECT/results/ego4d_multi_query_amortization.json}
EPOCHS=${EPOCHS:-20}
ROUTER_TYPE=${ROUTER_TYPE:-linear}
RESIDUAL_SCALE=${RESIDUAL_SCALE:-0.1}
ALIGNMENT_WEIGHT=${ALIGNMENT_WEIGHT:-0.0}
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun "$PYTHON" "$PROJECT/analyze_ego4d_multi_query.py" \
  --feature-dir "$FEATURE_DIR" \
  --train-annotation /dataset/Ego4D/v2/annotations/nlq_train.json \
  --val-annotation /dataset/Ego4D/v2/annotations/nlq_val.json \
  --text-cache "$TEXT_CACHE" \
  --epochs "$EPOCHS" \
  --router-type "$ROUTER_TYPE" \
  --residual-scale "$RESIDUAL_SCALE" \
  --alignment-weight "$ALIGNMENT_WEIGHT" \
  --output "$OUTPUT"
