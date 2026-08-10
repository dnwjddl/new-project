#!/bin/bash
#SBATCH --job-name=vft-prefix-pilot
#SBATCH --partition=gigabyte_a6000,suma_rtx4090,dell_rtx3090,base_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --exclude=node04,cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.out
#SBATCH --error=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.err

set -euo pipefail

PROJECT=/home/dnwjddl/project1_temporal_audit
PYTHON=/home/dnwjddl/miniconda3/envs/videoflextok/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun "$PYTHON" "$PROJECT/analyze_videoflextok_prefixes.py" \
  --feature-dir "$PROJECT/features/videoflextok_k600_pilot16k" \
  --conditions ordered reverse \
  --output "$PROJECT/results/ek100_videoflextok_prefix_roles_pilot16k.json"
