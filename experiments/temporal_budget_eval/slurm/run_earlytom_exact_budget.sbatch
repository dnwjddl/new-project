#!/bin/bash
#SBATCH --job-name=earlytom-exact
#SBATCH --partition=dell_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/dnwjddl/project1_budget_eval/logs/earlytom-exact-%A_%a.out
#SBATCH --error=/home/dnwjddl/project1_budget_eval/logs/earlytom-exact-%A_%a.err

set -euo pipefail

ROOT=/home/dnwjddl/project1_budget_eval
REPO="$ROOT/vendor/EarlyTom"
CODE="$ROOT/code"
BUDGETS=(4 8 16 32)
RATIOS=(0.001 0.001 0.0025 0.005)
BUDGET=${BUDGETS[$SLURM_ARRAY_TASK_ID]}
RETAIN_RATIO=${RATIOS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="earlytom-cap-k${BUDGET}"
OUTPUT="$ROOT/results/$RUN_NAME"
PRETRAIN=${PRETRAIN:-lmms-lab/llava-onevision-qwen2-0.5b-ov}
PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 1000) * 10 + SLURM_ARRAY_TASK_ID))

export HF_HOME="$ROOT/cache/huggingface"
export HF_TOKEN_PATH=/home/dnwjddl/.cache/huggingface/token
export HF_DATASETS_CACHE="$ROOT/cache/datasets"
export PIP_CACHE_DIR=/scratch/dnwjddl/pip-cache
export TMPDIR=/scratch/dnwjddl/tmp
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="$REPO:$REPO/LLaVA-NeXT:${PYTHONPATH:-}"
export WRAPPER=earlytom
export RETAIN_RATIO
export T=0.30
export M=6
export FIXED_VISUAL_TOKENS="$BUDGET"
export TEMPCOMPASS_DISABLE_GPT_JUDGE=1

mkdir -p "$OUTPUT"
cd "$REPO"

LIMIT_ARGS=()
if [[ -n "${LMMS_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LMMS_LIMIT")
fi

/scratch/dnwjddl/envs/earlytom/bin/python -m accelerate.commands.launch \
  --num_processes=1 \
  --main_process_port "$PORT" \
  -m lmms_eval \
  --model llava_onevision \
  --model_args pretrained="$PRETRAIN",conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
  --tasks tempcompass_multi_choice \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix "$RUN_NAME" \
  --output_path "$OUTPUT" \
  "${LIMIT_ARGS[@]}"

/scratch/dnwjddl/envs/earlytom/bin/python "$CODE/score_tempcompass_pairs.py" "$OUTPUT" \
  --budget "$BUDGET" \
  --output "$ROOT/scores/${RUN_NAME}.json"
