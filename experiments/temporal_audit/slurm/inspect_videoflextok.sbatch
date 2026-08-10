#!/bin/bash
#SBATCH --job-name=vft-contract
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.out
#SBATCH --error=/home/dnwjddl/project1_temporal_audit/logs/%x-%j.err

set -euo pipefail

PROJECT=/home/dnwjddl/project1_temporal_audit
PYTHON=/home/dnwjddl/miniconda3/envs/videoflextok/bin/python
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

srun "$PYTHON" "$PROJECT/inspect_videoflextok_latents.py"
