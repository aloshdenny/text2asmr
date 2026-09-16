#!/bin/bash
set -euo pipefail
while IFS= read -r -d '' kv; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done < /proc/1/environ
export PYTHONPATH=/workspace/text2asmr
export CLAP_LABEL_BASE=/workspace/clap_audios2
export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
export CLAP_MODEL=aoxo/clap-htsat-unfused-asmr
export CLAP_PROCESSOR=laion/clap-htsat-unfused
cd /workspace/text2asmr
exec python scripts/clap_label_audios2.py \
  --num-shards 12 --shard-index 0 \
  --clap-model aoxo/clap-htsat-unfused-asmr \
  --batch 128 --download-workers 16 --push-every 128
