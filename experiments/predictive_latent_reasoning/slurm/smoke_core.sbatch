#!/bin/bash
#SBATCH --job-name=plr-smoke
#SBATCH --partition=suma_rtx4090,asus_6000ada
#SBATCH --qos=base_qos
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/dnwjddl/predictive_latent_reasoning/logs/%x-%j.out
#SBATCH --error=/scratch/dnwjddl/predictive_latent_reasoning/logs/%x-%j.err

set -euo pipefail

ROOT=/scratch/dnwjddl/predictive_latent_reasoning
PYTHON=/home/dnwjddl/miniconda3/envs/ar_video/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun "$PYTHON" "$ROOT/code/train_core.py" \
  --cache "$ROOT/cache/ek100_vft_prequant_16slot.pt" \
  --output "$ROOT/results/smoke/full_cache_smoke.json" \
  --variant calibrated_correct \
  --seed 941002 \
  --epochs 1 \
  --batch-size 32 \
  --window 8 \
  --stride 12 \
  --width 96 \
  --memory-tokens 8 \
  --heads 6 \
  --correction-steps 1 \
  --num-workers 4 \
  --patience 1
