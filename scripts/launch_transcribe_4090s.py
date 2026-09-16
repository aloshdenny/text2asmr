#!/usr/bin/env python3
"""Launch N community RTX 4090 transcribe pods. Credentials from env only."""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runpod_launch import bootstrap_script, gql, list_gpus, create_pod  # noqa: E402


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    api_key = os.environ["RUNPOD_API_KEY"].strip()
    hf = os.environ["HF_TOKEN"].strip()
    pubkey = (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
    gpus = list_gpus(api_key)
    gpu = next(g for g in gpus if g["displayName"] == "RTX 4090")
    print(f"gpu id={gpu['id']} price={(gpu.get('lowestPrice') or {}).get('uninterruptablePrice')}")
    ids = []
    for i in range(n):
        args = Args(
            cloud="COMMUNITY",
            disk=40,
            container_disk=40,
            gpu_id=gpu["id"],
            name=f"t2a-transcribe-4090-{i}",
            image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        )
        env = {
            "HF_TOKEN": hf,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PUBLIC_KEY": pubkey,
            "TRANSCRIBE_BASE": "/workspace/t2a",
            "TEXT2ASMR_BOOTSTRAP": bootstrap_script(
                "transcribe", "HF_TOKEN",
                "https://github.com/aloshdenny/text2asmr",
                0, 8, 1, 0),
        }
        try:
            pod = create_pod(api_key, args, env)
        except SystemExit as e:
            print(f"shard {i} create failed: {e}")
            continue
        print(f"created shard {i} {pod}")
        ids.append(pod["id"])
        time.sleep(2)
    Path("/tmp/t2a_transcribe_pod_ids.json").write_text(json.dumps(ids))
    print("IDS", ids)
    return 0 if ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
