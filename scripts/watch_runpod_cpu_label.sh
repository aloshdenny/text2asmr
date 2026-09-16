#!/usr/bin/env bash
# Terminate a t2a label CPU pod if the annotator is dead for two checks.
# Usage: watch_runpod_cpu_label.sh POD_ID SSH_IP SSH_PORT [LOG]
# Does not print secrets. Requires ~/.t2a_runpod_env
set -euo pipefail
POD_ID="${1:?pod id}"
SSH_IP="${2:?ssh ip}"
SSH_PORT="${3:?ssh port}"
LOG="${4:-/Users/aoxo/vscode/text2asmr/label_tool/runpod_cpu_watch.log}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" "root@${SSH_IP}")
miss=0
say() { echo "[$(date -u +%H:%M:%S)] $POD_ID $*" | tee -a "$LOG"; }
# shellcheck disable=SC1090
source "$HOME/.t2a_runpod_env"
terminate() {
  python3 - "$POD_ID" <<'PY'
import os, sys, urllib.request
from pathlib import Path
for line in Path.home().joinpath(".t2a_runpod_env").read_text().splitlines():
    if line.startswith("export "):
        k,_,v=line[7:].partition("=")
        os.environ[k.strip()]=v.strip().strip("'").strip('"')
pid=sys.argv[1]
req=urllib.request.Request(
    f"https://rest.runpod.io/v1/pods/{pid}",
    method="DELETE",
    headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
             "User-Agent": "Mozilla/5.0 text2asmr-label-cpu"},
)
try:
    urllib.request.urlopen(req, timeout=30).read()
    print("terminated", pid)
except Exception as e:
    print("terminate_fail", type(e).__name__)
PY
}
say "watcher armed ssh=${SSH_IP}:${SSH_PORT}"
while true; do
  if "${SSH[@]}" 'pgrep -f annotate_audios2_gemini_batch.py >/dev/null'; then
    miss=0
    say "annotator alive"
  else
    miss=$((miss+1))
    say "annotator missing miss=$miss"
    if [ "$miss" -ge 2 ]; then
      say "terminating idle CPU pod $POD_ID"
      terminate
      exit 0
    fi
  fi
  sleep 300
done
