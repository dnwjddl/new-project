#!/bin/bash
#SBATCH --job-name=plr-cache
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/dnwjddl/predictive_latent_reasoning/logs/%x-%j.out
#SBATCH --error=/scratch/dnwjddl/predictive_latent_reasoning/logs/%x-%j.err

set -euo pipefail

ROOT=/scratch/dnwjddl/predictive_latent_reasoning
CODE=$ROOT/code
PYTHON=/home/dnwjddl/miniconda3/envs/ar_video/bin/python
mkdir -p "$ROOT/cache" "$ROOT/logs"

srun "$PYTHON" "$CODE/prepare_ek100_cache.py" \
  --feature-dir /home/dnwjddl/project1_temporal_audit/features/videoflextok_k600_full_temporal \
  --annotation-root /dataset/EK100/epic-kitchens-100-annotations \
  --output "$ROOT/cache/ek100_vft_prequant_16slot.pt" \
  --overwrite
