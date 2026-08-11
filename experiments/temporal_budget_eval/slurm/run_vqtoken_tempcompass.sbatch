#!/bin/bash
#SBATCH --job-name=vqtoken-tempcompass
#SBATCH --partition=dell_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/dnwjddl/project1_budget_eval/logs/vqtoken-tempcompass-%A_%a.out
#SBATCH --error=/home/dnwjddl/project1_budget_eval/logs/vqtoken-tempcompass-%A_%a.err

set -euo pipefail

ROOT=/home/dnwjddl/project1_budget_eval
REPO="$ROOT/vendor/VQToken"
CODE="$ROOT/code"
BUDGETS=(4 8 16 32)
BUDGET=${VQTOKEN_BUDGET:-${BUDGETS[$SLURM_ARRAY_TASK_ID]}}
METHOD=${VIDEO_TOKEN_BASELINE:-kmeans}
if [[ "$METHOD" == "kmeans" ]]; then
  RUN_NAME="vqtoken-k${BUDGET}"
else
  RUN_NAME="${METHOD}-k${BUDGET}"
fi
OUTPUT="$ROOT/results/$RUN_NAME"
PRETRAIN=${PRETRAIN:-lmms-lab/llava-onevision-qwen2-0.5b-ov}
PORT=${MAIN_PROCESS_PORT:-$((20000 + (SLURM_ARRAY_JOB_ID % 1000) * 10 + SLURM_ARRAY_TASK_ID))}

source /home/dnwjddl/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/dnwjddl/envs/vqtoken

export HF_HOME="$ROOT/cache/huggingface"
export HF_TOKEN_PATH=/home/dnwjddl/.cache/huggingface/token
export HF_DATASETS_CACHE="$ROOT/cache/datasets"
export PIP_CACHE_DIR=/scratch/dnwjddl/pip-cache
export TMPDIR=/scratch/dnwjddl/tmp
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export VQTOKEN_FIXED_BUDGET="$BUDGET"
export VIDEO_TOKEN_BASELINE="$METHOD"
export TEMPCOMPASS_DISABLE_GPT_JUDGE=1

mkdir -p "$OUTPUT"
cd "$REPO"

LIMIT_ARGS=()
if [[ -n "${LMMS_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LMMS_LIMIT")
fi

accelerate launch --num_processes=1 --main_process_port "$PORT" \
  -m lmms_eval \
  --model llava_onevision_vqtoken \
  --model_args pretrained="$PRETRAIN",conv_template=qwen_1_5,model_name=llava_qwen \
  --tasks tempcompass_multi_choice \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix "$RUN_NAME" \
  --output_path "$OUTPUT" \
  "${LIMIT_ARGS[@]}"

python "$CODE/score_tempcompass_pairs.py" "$OUTPUT" \
  --budget "$BUDGET" \
  --output "$ROOT/scores/${RUN_NAME}.json"
