#!/bin/bash
#SBATCH --job-name=g4-vm2-smoke
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
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODEL=${MODEL:-$ROOT/models/gemma-4-E2B-it}
MODEL_TAG=$(basename "$MODEL")
FRAMES=${FRAMES:-8}
SAMPLES=${SAMPLES:-3}

$ROOT/env/bin/python $ROOT/code/eval_videommev2.py \
  --model "$MODEL" \
  --parquet $ROOT/data/Video-MME-v2/test.parquet \
  --video-root $ROOT/data/Video-MME-v2/videos_extracted \
  --output $ROOT/results/$MODEL_TAG/videommev2_smoke_${FRAMES}f.jsonl \
  --num-frames "$FRAMES" \
  --max-samples "$SAMPLES"
