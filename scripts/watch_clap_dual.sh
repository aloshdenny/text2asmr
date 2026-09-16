#!/bin/bash
# Watch Tinkerspace (free) + RunPod (billed). Terminate the pod on done/idle.
set -euo pipefail
# shellcheck disable=SC1090
source "$HOME/.t2a_env"
LOG="${CLAP_WATCH_LOG:-$HOME/vscode/text2asmr/label_tool/watch_clap_dual.log}"
mkdir -p "$(dirname "$LOG")"
RP_HOST="${RP_HOST:-root@103.196.86.56}"
RP_PORT="${RP_PORT:-18495}"
TS_HOST="${TS_HOST:-tinkerspace@100.102.149.78}"
POD_ID="${POD_ID:-vjygklshggl753}"
IDLE_POLLS=0
ETA_SECS="${ETA_SECS:-10800}"  # 3h default; recalc after first progress
STARTED=$(date +%s)

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

terminate_pod() {
  python3 - <<PY
import json, os, urllib.request
key = os.environ["RUNPOD_API_KEY"]
pod = "${POD_ID}"
q = f'mutation {{ podTerminate(input: {{podId: "{pod}"}}) }}'
url = "https://api.runpod.io/graphql?api_key=" + key
req = urllib.request.Request(url, data=json.dumps({"query": q}).encode(),
    headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 t2a-clap-watch"})
print(urllib.request.urlopen(req, timeout=30).read().decode())
PY
}

rp() { ssh -o ConnectTimeout=12 -o StrictHostKeyChecking=no -p "$RP_PORT" "$RP_HOST" "$1"; }
ts() { ssh -o ConnectTimeout=12 "$TS_HOST" "$1"; }

log "WATCH start pod=$POD_ID eta_s=$ETA_SECS"
while true; do
  now=$(date +%s)
  rp_out=$(rp 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; echo ---PS---; ps -eo pid,cmd | grep "[c]lap_label_audios2"; echo ---LOG---; grep -E "CLAP_DONE|gpu batch=|accepted=|commit ok" /workspace/clap_audios2/clap_label.log /workspace/clap_label.log 2>/dev/null | tail -n 8' || echo RP_SSH_FAIL)
  ts_out=$(ts 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; echo ---PS---; ps -eo pid,cmd | grep "[c]lap_label_audios2"; echo ---LOG---; tail -n 8 ~/t2a/clap_audios2/clap_label.log 2>/dev/null' || echo TS_SSH_FAIL)
  log "RUNPOD $rp_out"
  log "TINKER $ts_out"

  if echo "$rp_out" | grep -q "CLAP_DONE"; then
    log "RUNPOD done — terminate $POD_ID"
    terminate_pod || true
    log "TERMINATED"
    # keep watching tinkerspace only
    POD_ID=""
  fi

  if [ -n "$POD_ID" ]; then
    if echo "$rp_out" | grep -q "clap_label_audios2.py"; then
      IDLE_POLLS=0
    else
      util=$(echo "$rp_out" | head -n 1 | awk -F',' '{gsub(/[^0-9]/,"",$1); print $1+0}')
      if [ "${util:-0}" -lt 2 ]; then
        IDLE_POLLS=$((IDLE_POLLS + 1))
      else
        IDLE_POLLS=0
      fi
    fi
    if [ "$IDLE_POLLS" -ge 2 ]; then
      log "RUNPOD idle/no PID x2 — terminate $POD_ID"
      terminate_pod || true
      POD_ID=""
    fi
  fi

  if [ -z "$POD_ID" ]; then
    if echo "$ts_out" | grep -q "CLAP_DONE"; then
      log "BOTH done"
      exit 0
    fi
    if ! echo "$ts_out" | grep -q "clap_label_audios2.py"; then
      log "tinkerspace clap gone and runpod terminated"
      exit 0
    fi
  fi

  elapsed=$((now - STARTED))
  if [ "$elapsed" -ge "$ETA_SECS" ] && [ -n "$POD_ID" ]; then
    log "ETA alarm elapsed=${elapsed}s — still running, re-arm +3h"
    STARTED=$now
    ETA_SECS=10800
  fi
  sleep 180
done
