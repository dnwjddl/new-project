#!/bin/bash
#SBATCH --job-name=plr-gate
#SBATCH --partition=suma_rtx4090,asus_6000ada
#SBATCH --qos=base_qos
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/dnwjddl/predictive_latent_reasoning/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dnwjddl/predictive_latent_reasoning/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/scratch/dnwjddl/predictive_latent_reasoning
PYTHON=/home/dnwjddl/miniconda3/envs/ar_video/bin/python
VARIANTS=(observation_gate innovation_gate)
SEEDS=(941002 941003 941004)
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
VARIANT=${VARIANTS[$VARIANT_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun "$PYTHON" "$ROOT/code/train_core.py" \
  --cache "$ROOT/cache/ek100_vft_prequant_16slot.pt" \
  --output "$ROOT/results/${VARIANT}_seed${SEED}.json" \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --epochs 14 \
  --batch-size 32 \
  --window 12 \
  --stride 6 \
  --width 192 \
  --memory-tokens 8 \
  --correction-steps 1 \
  --gate-weight 0.001 \
  --patience 4
