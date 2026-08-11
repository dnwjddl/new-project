#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDEO_MME_ROOT:-/scratch/dnwjddl/gemma4_video_eval/data/Video-MME-v2}"
START="${1:-1}"
END="${2:-40}"
PARALLEL="${3:-8}"

download_one() {
  local number="$1"
  local archive
  archive="$(printf '%03d' "$number")"

  local final_path="${ROOT}/videos/${archive}.zip"
  local partial_path="${ROOT}/videos/${archive}.zip.part"
  local url="https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2/resolve/main/videos/${archive}.zip?download=true"

  if [[ -f "${final_path}" ]]; then
    echo "[skip] ${archive}.zip"
    return
  fi

  echo "[download] ${archive}.zip"
  wget --no-verbose --continue --output-document="${partial_path}" "${url}"
  unzip -tq "${partial_path}" >/dev/null
  mv "${partial_path}" "${final_path}"
  echo "[complete] ${archive}.zip"
}

export ROOT
export -f download_one

mkdir -p "${ROOT}/videos"
seq "${START}" "${END}" | xargs -n 1 -P "${PARALLEL}" bash -c 'download_one "$1"' _
