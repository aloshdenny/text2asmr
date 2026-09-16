#!/usr/bin/env python3
"""SSH-start CLAP label jobs on t2a-clap-* pods (bootstrap env is not auto-run)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

GQL = "https://api.runpod.io/graphql"
PREFIX = "t2a-clap-"
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "text2asmr" / "data"


def gql(query: str, key: str) -> dict:
    request = urllib.request.Request(
        f"{GQL}?api_key={key}",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 t2a-clap-kick"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(result["errors"])
    return result["data"]


def shard_from_name(name: str) -> int:
    return int(name.rsplit("-", 1)[-1])


def kick(host: str, shard: int, num_shards: int) -> bool:
    script_b64 = base64.b64encode((ROOT / "clap_label_audios2.py").read_bytes()).decode()
    ont_b64 = base64.b64encode((DATA / "ontology.py").read_bytes()).decode()
    seg_b64 = base64.b64encode((DATA / "segment.py").read_bytes()).decode()
    # Stream base64 in chunks via SSH stdin to avoid arg limits.
    remote = f"""
set -e
mkdir -p /workspace/text2asmr/scripts /workspace/text2asmr/text2asmr/data
if pgrep -f '[c]lap_label_audios2' >/dev/null; then echo ALREADY_RUNNING; exit 0; fi
python -m pip install -q -U pip
python -m pip install -q "huggingface_hub[hf_xet]" transformers soundfile numpy
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
printf '' > /workspace/text2asmr/text2asmr/__init__.py
printf '' > /workspace/text2asmr/text2asmr/data/__init__.py
echo {script_b64} | base64 -d > /workspace/text2asmr/scripts/clap_label_audios2.py
echo {ont_b64} | base64 -d > /workspace/text2asmr/text2asmr/data/ontology.py
echo {seg_b64} | base64 -d > /workspace/text2asmr/text2asmr/data/segment.py
export PYTHONPATH=/workspace/text2asmr
export CLAP_LABEL_BASE=/workspace/clap_audios2
export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
nohup python /workspace/text2asmr/scripts/clap_label_audios2.py \\
  --num-shards {num_shards} --shard-index {shard} \\
  --batch 32 --download-workers 8 --push-every 48 \\
  > /workspace/clap_label.log 2>&1 &
echo KICKED_PID=$!
sleep 2
pgrep -af clap_label_audios2 || true
echo KICK_DONE
exit
"""
    for attempt in range(8):
        try:
            z = subprocess.run(
                [
                    "ssh", "-tt", "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15",
                    "-i", str(Path.home() / ".ssh" / "id_ed25519"),
                    f"{host}@ssh.runpod.io",
                ],
                input=remote, text=True, capture_output=True, timeout=180,
            )
            if "KICK_DONE" in z.stdout or "ALREADY_RUNNING" in z.stdout:
                print(f"KICKED shard={shard} host={host}", flush=True)
                return True
            print(f"RETRY shard={shard} attempt={attempt} out={z.stdout[-400:]}", flush=True)
        except Exception as exc:
            print(f"RETRY shard={shard} attempt={attempt} err={exc}", flush=True)
        time.sleep(15)
    return False


def main() -> int:
    key = os.environ["RUNPOD_API_KEY"]
    state = json.loads(Path("/tmp/t2a_clap_fleet.json").read_text()) if Path("/tmp/t2a_clap_fleet.json").exists() else {}
    num_shards = int(state.get("shards") or 12)
    pods = gql(
        """query { myself { pods { id name machine { podHostId } } } }""",
        key,
    )["myself"]["pods"] or []
    pods = [p for p in pods if (p.get("name") or "").startswith(PREFIX)]
    print(f"live={[(p['name'], p['id']) for p in pods]}", flush=True)
    ok = 0
    for p in pods:
        host = (p.get("machine") or {}).get("podHostId")
        if not host:
            # wait for runtime
            for _ in range(20):
                time.sleep(10)
                pods2 = gql(
                    """query { myself { pods { id name machine { podHostId } } } }""",
                    key,
                )["myself"]["pods"] or []
                match = next((x for x in pods2 if x["id"] == p["id"]), None)
                host = ((match or {}).get("machine") or {}).get("podHostId")
                if host:
                    break
        if not host:
            print(f"NO_HOST {p['name']}", flush=True)
            continue
        if kick(host, shard_from_name(p["name"]), num_shards):
            ok += 1
    print(f"KICKED {ok}/{len(pods)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
