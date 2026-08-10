#!/bin/bash
#SBATCH --job-name=vft-prefix-probe
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.out
#SBATCH --error=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.err

set -euo pipefail

PROJECT=/home/dnwjddl/project1_temporal_audit
PYTHON=/home/dnwjddl/miniconda3/envs/videoflextok/bin/python
export PYTHONUNBUFFERED=1

srun "$PYTHON" "$PROJECT/analyze_videoflextok_prefixes.py" \
  --feature-dir "$PROJECT/features/videoflextok_k600_17f" \
  --output "$PROJECT/results/ek100_videoflextok_prefix_roles.json"
