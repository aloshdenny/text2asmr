#!/usr/bin/env bash
# Restart Vertex labelling + Hub sync whenever they die.
# Exit only after GEMINI_ALL_DONE so a container restart picks this up via READY.
set -u
ROOT="${T2A_ROOT:-/workspace/text2asmr}"
cd "$ROOT" || exit 1

exec 9>/workspace/supervise.lock
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "[$(date -u +%FT%TZ)] supervise already running"
    exit 0
  fi
elif ! mkdir /workspace/supervise.lock.dir 2>/dev/null; then
  echo "[$(date -u +%FT%TZ)] supervise already running"
  exit 0
fi

# RunPod injects env on PID 1; SSH/cron/nohup often do not inherit it.
if [ -r /proc/1/environ ]; then
  while IFS= read -r kv; do
    case "$kv" in
      HF_TOKEN=*|HF_TOKENS=*|HUGGING_FACE_HUB_TOKEN=*|GEMINI_*|GOOGLE_*|HF_XET_*|HUB_SYNC_*)
        export "$kv"
        ;;
    esac
  done < <(tr '\0' '\n' < /proc/1/environ)
fi
# shellcheck disable=SC1091
source /workspace/t2a_pod.env 2>/dev/null || true
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GEMINI_BACKEND="${GEMINI_BACKEND:-vertex}"
export TRANSCRIBE_BASE="${TRANSCRIBE_BASE:-/workspace/t2a}"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/workspace/adc.json}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CACHE="${HF_XET_CACHE:-/workspace/hf-xet-cache}"
export GEMINI_HUB_SEM="${GEMINI_HUB_SEM:-64}"
export HF_MAX_REQ_PER_5MIN="${HF_MAX_REQ_PER_5MIN:-980}"
export GEMINI_MAX_INFLIGHT="${GEMINI_MAX_INFLIGHT:-80}"
PY="${T2A_PYTHON:-/workspace/venv/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python)"
fi
LOG=/workspace/supervise.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

mkdir -p "$TRANSCRIBE_BASE" "$HF_XET_CACHE" "$ROOT/label_tool" /workspace/t2a
export DEBIAN_FRONTEND=noninteractive
if ! command -v ffmpeg >/dev/null 2>&1; then
  say "installing ffmpeg"
  apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
fi

gemini_alive() { pgrep -f '[a]nnotate_audios2_gemini_batch.py' >/dev/null; }
sync_alive() { pgrep -f '[s]ync_clap_ft_manifest.sh' >/dev/null; }

start_gemini() {
  say "starting gemini annotator"
  nohup "$PY" -u scripts/annotate_audios2_gemini_batch.py \
    --until-empty --poll --poll-sec 45 --stripe-models \
    --collect-workers "${GEMINI_COLLECT_WORKERS:-48}" \
    --max-inflight "${GEMINI_MAX_INFLIGHT}" \
    --batch-size "${GEMINI_BATCH_SIZE:-80}" \
    --file-list label_tool/audios2_file_list.json \
    >> /workspace/gemini_vertex_batch.log 2>&1 &
}

start_sync() {
  say "starting hub sync"
  nohup bash scripts/sync_clap_ft_manifest.sh "${HUB_SYNC_INTERVAL:-90}" \
    >> /workspace/clap_ft_sync.log 2>&1 &
}

say "supervise up root=$ROOT py=$PY"
while true; do
  if [ -f "$ROOT/label_tool/GEMINI_ALL_DONE" ]; then
    say "GEMINI_ALL_DONE present; holding container for watcher pull"
    # Keep the pod process tree alive until the Mac watcher terminates it.
    while true; do sleep 3600; done
  fi
  if ! gemini_alive; then
    say "gemini dead; restarting"
    start_gemini
  fi
  if ! sync_alive; then
    say "sync dead; restarting"
    start_sync
  fi
  sleep 15
done
