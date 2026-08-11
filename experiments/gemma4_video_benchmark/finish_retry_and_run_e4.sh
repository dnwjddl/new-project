#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scratch/dnwjddl/gemma4_video_eval}"
FIRST_JOB="${1:?first retry job id is required}"
SECOND_JOB="${2:?second retry job id is required}"
LOG="${ROOT}/logs/videommev2_retry_followup.log"

exec >>"${LOG}" 2>&1
echo "[$(date -Is)] waiting for retry jobs ${FIRST_JOB},${SECOND_JOB}"
while [[ -n "$(squeue -h -j "${FIRST_JOB},${SECOND_JOB}")" ]]; do
  sleep 60
done

SUMMARY="${ROOT}/results/gemma-4-E2B-it/videommev2_64f_summary.json"
"${ROOT}/env/bin/python" "${ROOT}/code/summarize_videommev2.py" \
  "${ROOT}/results/gemma-4-E2B-it"/videommev2_64f_shard*.jsonl \
  --output "${SUMMARY}"

"${ROOT}/env/bin/python" - "${SUMMARY}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
assert summary["questions"] == 3200, summary
assert summary["complete_groups"] == 800, summary
assert summary["errors"] == 0, summary
print(
    "E2B retry validated:",
    f"accuracy={summary['accuracy']:.4f}",
    f"limit_hits={summary['generation_limit_hits']}",
)
PY

echo "[$(date -Is)] starting clean E4B run"
MODELS=gemma-4-E4B-it "${ROOT}/code/run_videommev2_pipeline.sh"
