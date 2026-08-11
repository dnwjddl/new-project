#!/bin/bash
#SBATCH --job-name=g4-videommev2
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
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

: "${MODEL:?Submit with --export=ALL,MODEL=/path/to/model}"
FRAMES=${FRAMES:-64}
SHARDS=${SHARDS:-8}
MODEL_TAG=$(basename "$MODEL")
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
EXTRA_ARGS=()
if [[ "${RETRY_NONCOMPLIANT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--retry-noncompliant)
fi

$ROOT/env/bin/python $ROOT/code/eval_videommev2.py \
  --model "$MODEL" \
  --parquet $ROOT/data/Video-MME-v2/test.parquet \
  --video-root $ROOT/data/Video-MME-v2/videos_extracted \
  --output $ROOT/results/$MODEL_TAG/videommev2_${FRAMES}f_shard${SLURM_ARRAY_TASK_ID}.jsonl \
  --num-frames "$FRAMES" \
  --rank "$SLURM_ARRAY_TASK_ID" \
  --world-size "$SHARDS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${EXTRA_ARGS[@]}"
