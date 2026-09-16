#!/bin/bash
set -euo pipefail
# shellcheck disable=SC1090
source "$HOME/.t2a_env"
export PYTHONPATH="$HOME/t2a/text2asmr"
export CLAP_LABEL_BASE="$HOME/t2a/clap_audios2"
export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
export CLAP_MODEL="${CLAP_MODEL:-aoxo/clap-htsat-unfused-asmr}"
export CLAP_PROCESSOR="${CLAP_PROCESSOR:-laion/clap-htsat-unfused}"
cd "$HOME/t2a/text2asmr"
PY="${HOME}/t2a/clap-venv/bin/python"
if [ ! -x "$PY" ]; then
  PY=python3
fi
exec "$PY" scripts/clap_label_audios2.py \
  --num-shards 12 --shard-index "${CLAP_SHARD:-1}" \
  --clap-model "$CLAP_MODEL" \
  --batch "${CLAP_BATCH:-64}" \
  --download-workers 8 \
  --push-every 128
