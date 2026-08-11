#!/bin/bash
#SBATCH --job-name=g4-tempcompass
#SBATCH --partition=suma_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/dnwjddl/gemma4_video_eval/logs/%x_%A_%a.out
#SBATCH --error=/scratch/dnwjddl/gemma4_video_eval/logs/%x_%A_%a.err

set -euo pipefail

ROOT=/scratch/dnwjddl/gemma4_video_eval
export HF_HOME=/scratch/dnwjddl/hf
export TRANSFORMERS_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

: "${MODEL:?Submit with --export=ALL,MODEL=/path/to/model}"
FRAMES=${FRAMES:-32}
SHARDS=${SHARDS:-8}
MODEL_TAG=$(basename "$MODEL")

$ROOT/env/bin/python $ROOT/code/eval_tempcompass.py \
  --model "$MODEL" \
  --data-root /scratch/dnwjddl/project1_budget_eval/data/TempCompass \
  --video-root $ROOT/data/TempCompass/videos \
  --output $ROOT/results/$MODEL_TAG/tempcompass_${FRAMES}f_shard${SLURM_ARRAY_TASK_ID}.jsonl \
  --num-frames "$FRAMES" \
  --rank "$SLURM_ARRAY_TASK_ID" \
  --world-size "$SHARDS"
