#!/bin/bash
# Restart the whisper chain if the Python job dies before both repos finish.
set -u
LOG="${HOME}/t2a/whisper_chain.log"
KICK="${HOME}/t2a/text2asmr/scripts/kick_transcribe_tinkerspace.sh"

alive() {
  pgrep -f '/t2a/.venv-transcribe/bin/python scripts/transcribe_audios2.py' >/dev/null
}

start_whisper() {
  export TMUX_TMPDIR=/tmp
  tmux start-server
  tmux kill-session -t whisper 2>/dev/null || true
  tmux new-session -d -s whisper -- /bin/bash -lc \
    "source ~/.t2a_env; export PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 CUDA_VISIBLE_DEVICES=0; exec ${KICK} >> ${LOG} 2>&1"
}

while true; do
  if grep -q 'AUDIOS3_DONE' "$LOG" 2>/dev/null; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] both repos done"
    exit 0
  fi
  if ! alive; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] whisper dead; restarting"
    start_whisper
  fi
  sleep 120
done
