#!/usr/bin/env bash
# Terminate GCE t2a-label VM if annotator is dead twice or job logs DONE.
# Usage: watch_gce_t2a_label.sh [ZONE]
set -euo pipefail
PROJ="${GCP_PROJECT:-project-9da5a2fe-3df4-485e-9a9}"
ZONE="${1:-us-east1-b}"
NAME="t2a-label"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=12 aoxo@35.185.77.101)
LOG=/Users/aoxo/vscode/text2asmr/label_tool/gce_t2a_watch.log
miss=0
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
terminate() {
  gcloud compute instances delete "$NAME" --project="$PROJ" --zone="$ZONE" --quiet || true
  say "deleted $NAME"
}
say "watcher armed zone=$ZONE"
while true; do
  if "${SSH[@]}" 'pgrep -f annotate_audios2_gemini_batch.py >/dev/null'; then
    miss=0
    say "annotator alive"
    if "${SSH[@]}" 'grep -q "batch annotate done" /opt/text2asmr/label_tool/gemini_vertex_batch.gce.log'; then
      say "DONE marker — pulling ledgers then terminate"
      rsync -az aoxo@35.185.77.101:/opt/text2asmr/label_tool/gemini_audios2.batch.vertex.jsonl \
        /Users/aoxo/vscode/text2asmr/label_tool/gemini_audios2.batch.vertex.gce.jsonl || true
      terminate
      exit 0
    fi
  else
    miss=$((miss+1))
    say "annotator missing miss=$miss"
    if [ "$miss" -ge 2 ]; then
      say "idle VM — terminate"
      terminate
      exit 0
    fi
  fi
  sleep 300
done
