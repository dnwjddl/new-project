#!/bin/bash
#SBATCH --job-name=ek100-probe
#SBATCH --partition=gigabyte_a6000,suma_rtx4090,dell_rtx3090,base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --exclude=node04,cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.out
#SBATCH --error=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.err

set -euo pipefail

PROJECT=/home/dnwjddl/project1_temporal_audit
PYTHON=/home/dnwjddl/miniconda3/envs/ar_video/bin/python
export PYTHONUNBUFFERED=1

srun "$PYTHON" "$PROJECT/train_linear_probes.py" \
  --feature-dir "$PROJECT/features/viclip_l14_8f" \
  --output "$PROJECT/results/ek100_viclip_temporal_audit.json"
