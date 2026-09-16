#!/bin/bash
# Restart Vertex labelling + clap-ft-data sync if they die.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1090
source "$HOME/.t2a_env" 2>/dev/null || true
export PYTHONUNBUFFERED=1 GEMINI_BACKEND=vertex GEMINI_HUB_SEM=8
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOG="$ROOT/label_tool/gemini_and_sync_watch.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

gemini_alive() { pgrep -f '[P]ython -u scripts/annotate_audios2_gemini_batch.py' >/dev/null; }
sync_alive() { pgrep -f '[s]cripts/sync_clap_ft_manifest.sh' >/dev/null; }

start_gemini() {
  nohup python3 -u scripts/annotate_audios2_gemini_batch.py \
    --until-empty --poll --poll-sec 45 --stripe-models \
    --collect-workers 8 --max-inflight 40 --batch-size 80 \
    --file-list label_tool/audios2_file_list.json \
    >> label_tool/gemini_vertex_batch.mac.log 2>&1 &
}

start_sync() {
  nohup bash scripts/sync_clap_ft_manifest.sh 600 \
    >> label_tool/clap_ft_sync.log 2>&1 &
}

say "watcher up gemini=$(gemini_alive && echo yes || echo no) sync=$(sync_alive && echo yes || echo no)"
while true; do
  sleep 120
  if ! gemini_alive; then
    say "gemini dead; restarting"
    start_gemini
  fi
  if ! sync_alive; then
    say "sync dead; restarting"
    start_sync
  fi
done
