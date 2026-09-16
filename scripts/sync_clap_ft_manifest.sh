#!/usr/bin/env bash
# Rebuild clap_finetune_manifest.jsonl from latest Gemini clap_train rows and
# upload to HF so the RunPod trainer can ingest mid-run.
# One Hub commit per cycle (manifest + live Vertex ledgers) to saturate
# useful sync without burning the per-token REST budget on tiny commits.
# Transient network/DNS failures must not kill the loop.
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
source ~/.t2a_env 2>/dev/null || true
# shellcheck disable=SC1091
source /workspace/t2a_pod.env 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
PY="${T2A_PYTHON:-python3}"
INTERVAL="${1:-90}"
REPO="${CLAP_FT_DATA_REPO:-aoxo/clap-ft-data}"
FILE=clap_finetune_manifest.jsonl

echo "sync_loop interval=${INTERVAL}s repo=${REPO} py=$PY"
while true; do
  if ! "$PY" scripts/build_clap_finetune_manifest.py --out "label_tool/${FILE}"; then
    echo "[$(date -u +%FT%TZ)] build failed; retry in ${INTERVAL}s" >&2
    sleep "$INTERVAL"
    continue
  fi
  if ! "$PY" - <<PY
import os, time
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

repo = "${REPO}"
api = HfApi(token=os.environ["HF_TOKEN"])
root = Path("label_tool")
pairs = [
    (root / "${FILE}", "${FILE}"),
    (root / "gemini_audios2.batch.vertex.jsonl", "gemini_audios2.batch.vertex.jsonl"),
    (root / "clap_train_audios2.batch.vertex.jsonl", "clap_train_audios2.batch.vertex.jsonl"),
]
ops = []
for local, remote in pairs:
    if local.is_file() and local.stat().st_size > 0:
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
if not ops:
    print("no files to upload", flush=True)
    raise SystemExit(0)
manifest = root / "${FILE}"
n = sum(1 for _ in open(manifest)) if manifest.is_file() else 0
# Never clobber Hub with a cold-start empty rebuild (pod has uid set, not full clap_train ledgers).
if n < 200000:
    print("skip upload; manifest rows", n, "< 200000 guard", flush=True)
    raise SystemExit(0)
for attempt in range(5):
    try:
        api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
        api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            operations=ops,
            commit_message="gemini label sync",
        )
        n = sum(1 for _ in open(root / "${FILE}"))
        print("uploaded commit files", len(ops), "manifest_rows", n, flush=True)
        break
    except Exception as exc:
        print(f"upload fail attempt={attempt+1}: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(5 * (attempt + 1))
else:
    raise SystemExit(1)
PY
  then
    echo "[$(date -u +%FT%TZ)] upload failed; will retry next cycle" >&2
  fi
  sleep "$INTERVAL"
done
