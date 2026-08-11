#!/bin/bash
#SBATCH --job-name=earlytom-tempcompass
#SBATCH --partition=dell_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dnwjddl/project1_budget_eval/logs/earlytom-tempcompass-%j.out
#SBATCH --error=/home/dnwjddl/project1_budget_eval/logs/earlytom-tempcompass-%j.err

set -euo pipefail

ROOT=/home/dnwjddl/project1_budget_eval
REPO="$ROOT/vendor/EarlyTom"
CODE="$ROOT/code"
RETAIN_RATIO=${RETAIN_RATIO:-0.10}
T=${T:-0.30}
RUN_NAME=${RUN_NAME:-earlytom-r${RETAIN_RATIO}}
OUTPUT="$ROOT/results/$RUN_NAME"
PRETRAIN=${PRETRAIN:-lmms-lab/llava-onevision-qwen2-0.5b-ov}

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
export T
export M=6
export TEMPCOMPASS_DISABLE_GPT_JUDGE=1

mkdir -p "$OUTPUT"
cd "$REPO"

LIMIT_ARGS=()
if [[ -n "${LMMS_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LMMS_LIMIT")
fi

/scratch/dnwjddl/envs/earlytom/bin/python -m accelerate.commands.launch \
  --num_processes=1 \
  --main_process_port "${MAIN_PROCESS_PORT:-29600}" \
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
  --output "$ROOT/scores/${RUN_NAME}.json"
