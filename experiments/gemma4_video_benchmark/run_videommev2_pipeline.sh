#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scratch/dnwjddl/gemma4_video_eval}"
ARCHIVES="${ROOT}/data/Video-MME-v2/videos"
VIDEOS="${ROOT}/data/Video-MME-v2/videos_extracted"
LOG="${ROOT}/logs/videommev2_pipeline.log"

exec >>"${LOG}" 2>&1
echo "[$(date -Is)] Waiting for all Video-MME-v2 videos"

while true; do
  archive_count="$(find "${ARCHIVES}" -maxdepth 1 -name '*.zip' -type f | wc -l)"
  video_count="$(find "${VIDEOS}" -name '*.mp4' -type f | wc -l)"
  echo "[$(date -Is)] archives=${archive_count}/40 videos=${video_count}/800"

  if [[ "${video_count}" -ge 800 ]]; then
    break
  fi

  if [[ "${archive_count}" -ge 40 ]] && ! pgrep -f extract_videommev2.py >/dev/null; then
    "${ROOT}/env/bin/python" "${ROOT}/code/extract_videommev2.py" \
      --archive-root "${ARCHIVES}" \
      --output "${VIDEOS}" \
      --expected-archives 40 \
      --once
  fi
  sleep 60
done

wait_for_jobs() {
  local first="$1"
  local second="$2"
  while [[ -n "$(squeue -h -j "${first},${second}")" ]]; do
    echo "[$(date -Is)] waiting for jobs ${first},${second}"
    sleep 60
  done
}

validate_summary() {
  local summary="$1"
  "${ROOT}/env/bin/python" - "${summary}" <<'PY'
import json
import sys

path = sys.argv[1]
summary = json.load(open(path))
assert summary["questions"] == 3200, summary
assert summary["complete_groups"] == 800, summary
assert summary["errors"] == 0, summary
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

run_model() {
  local model_tag="$1"
  local model="${ROOT}/models/${model_tag}"
  local result_root="${ROOT}/results/${model_tag}"
  mkdir -p "${result_root}"

  echo "[$(date -Is)] submitting ${model_tag}"
  local first second
  first="$(sbatch --parsable --job-name="g4-vmm-${model_tag}-a" \
    --array=0-3 --partition=base_suma_rtx3090 \
    --export="ALL,MODEL=${model}" \
    "${ROOT}/code/run_videommev2_array.sbatch")"
  second="$(sbatch --parsable --job-name="g4-vmm-${model_tag}-b" \
    --array=4-7 --partition=dell_rtx3090 \
    --export="ALL,MODEL=${model}" \
    "${ROOT}/code/run_videommev2_array.sbatch")"
  first="${first%%;*}"
  second="${second%%;*}"
  echo "[$(date -Is)] submitted jobs ${first},${second}"

  wait_for_jobs "${first}" "${second}"

  "${ROOT}/env/bin/python" "${ROOT}/code/summarize_videommev2.py" \
    "${result_root}"/videommev2_64f_shard*.jsonl \
    --output "${result_root}/videommev2_64f_summary.json"
  validate_summary "${result_root}/videommev2_64f_summary.json"
  echo "[$(date -Is)] completed ${model_tag}"
}

MODELS=${MODELS:-"gemma-4-E2B-it gemma-4-E4B-it"}
for model_tag in ${MODELS}; do
  run_model "${model_tag}"
done
touch "${ROOT}/results/videommev2_pipeline.complete"
echo "[$(date -Is)] pipeline complete"
