#!/bin/bash
#SBATCH --job-name=g4-tc-smoke
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/dnwjddl/gemma4_video_eval/logs/%x_%j.out
#SBATCH --error=/scratch/dnwjddl/gemma4_video_eval/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/dnwjddl/gemma4_video_eval
export HF_HOME=/scratch/dnwjddl/hf
export TRANSFORMERS_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODEL=${MODEL:-$ROOT/models/gemma-4-E2B-it}
MODEL_TAG=$(basename "$MODEL")

$ROOT/env/bin/python $ROOT/code/eval_tempcompass.py \
  --model "$MODEL" \
  --data-root /scratch/dnwjddl/project1_budget_eval/data/TempCompass \
  --video-root $ROOT/data/TempCompass/videos \
  --output $ROOT/results/$MODEL_TAG/tempcompass_smoke.jsonl \
  --num-frames 8 \
  --max-samples 3
