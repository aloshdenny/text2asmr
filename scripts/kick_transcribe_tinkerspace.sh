#!/bin/bash
set -euo pipefail
source "$HOME/.t2a_env"
export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
export TRANSCRIBE_BASE="${TRANSCRIBE_BASE:-$HOME/t2a}"
cd "$HOME/t2a/text2asmr"
PY="${HOME}/t2a/.venv-transcribe/bin/python"
if [ ! -x "$PY" ]; then
  echo "missing $PY" >&2
  exit 1
fi

run_repo() {
  local repo="$1"
  local attempt=1
  while true; do
    echo "========== TRANSCRIBE $repo $(date -u +%Y-%m-%dT%H:%M:%SZ) attempt=$attempt =========="
    if "$PY" scripts/transcribe_audios2.py \
      --repo "$repo" \
      --model large-v3 \
      --compute-type float16 \
      --transcribe-workers 3 \
      --producer-workers 16 \
      --uploader-workers 1 \
      --upload-batch-size 128 \
      --upload-batch-timeout 90
    then
      echo "========== $repo OK $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
      return 0
    fi
    echo "========== $repo CRASH exit=$? $(date -u +%Y-%m-%dT%H:%M:%SZ); retry in 30s =========="
    attempt=$((attempt + 1))
    sleep 30
  done
}

run_repo aoxo/audios2
echo "AUDIOS2_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_repo aoxo/audios3
echo "AUDIOS3_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
