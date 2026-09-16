#!/usr/bin/env python3
"""Launch a disposable, non-overlapping RunPod transcription fleet."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

GQL = "https://api.runpod.io/graphql"
GPU_ID = "NVIDIA GeForce RTX 4090"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CLOUD_TYPE = "COMMUNITY"


def gql(query: str, key: str) -> dict:
    request = urllib.request.Request(
        f"{GQL}?api_key={key}",
        data=json.dumps({"query": query}).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 text2asmr-fleet",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(result["errors"])
    return result["data"]


def bootstrap(
    script_b64: str,
    creators_b64: str,
    shards: int,
    index: int,
    include_creators: bool,
) -> str:
    creator_flag = "--creators-file" if include_creators else "--exclude-creators-file"
    return textwrap.dedent(
        f"""
        set -euo pipefail
        exec > >(tee -a /workspace/bootstrap.log) 2>&1
        echo "BOOT shard={index}/{shards} $(date -u)"
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq ffmpeg >/dev/null
        python -m pip install -q faster-whisper "huggingface_hub[hf_xet]"
        mkdir -p /workspace/t2a
        echo {script_b64} | base64 -d > /workspace/transcribe_audios2.py
        echo {creators_b64} | base64 -d > /workspace/tinkerspace_creators.txt
        export TRANSCRIBE_BASE=/workspace/t2a
        export PYTHONUNBUFFERED=1
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        python /workspace/transcribe_audios2.py \\
          --model large-v3 --compute-type float16 \\
          --transcribe-workers 3 --producer-workers 3 --uploader-workers 1 \\
          --upload-batch-size 128 \\
          {creator_flag} /workspace/tinkerspace_creators.txt \\
          --num-shards {shards} --shard-index {index} --shard-key file
        echo "TRANSCRIBE_DONE shard={index}"
        """
    ).strip()


def create_one(
    key: str,
    token: str,
    public_key: str,
    script_b64: str,
    creators_b64: str,
    shards: int,
    index: int,
    include_creators: bool,
    pod_prefix: str,
) -> dict:
    env = {
        "HF_TOKEN": token,
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PUBLIC_KEY": public_key,
        "TEXT2ASMR_BOOTSTRAP": bootstrap(
            script_b64, creators_b64, shards, index, include_creators
        ),
    }
    env_gql = ", ".join(
        f'{{key:{json.dumps(k)},value:{json.dumps(v)}}}' for k, v in env.items()
    )
    query = f"""
    mutation {{
      podFindAndDeployOnDemand(input: {{
        cloudType: {CLOUD_TYPE}
        gpuCount: 1
        volumeInGb: 20
        containerDiskInGb: 30
        minVcpuCount: 4
        minMemoryInGb: 24
        gpuTypeId: {json.dumps(GPU_ID)}
        name: {json.dumps(f"{pod_prefix}{index:02d}")}
        imageName: {json.dumps(IMAGE)}
        dockerArgs: ""
        ports: "22/tcp"
        volumeMountPath: "/workspace"
        env: [{env_gql}]
      }}) {{ id name costPerHr machineId }}
    }}
    """
    return gql(query, key)["podFindAndDeployOnDemand"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="?", type=int, default=60)
    parser.add_argument(
        "--include-creators",
        action="store_true",
        help="process only transcribe_shards/tinkerspace.txt instead of excluding it",
    )
    parser.add_argument(
        "--shard-indices",
        help="comma-separated subset to launch; defaults to every shard",
    )
    parser.add_argument("--pod-prefix", default="t2a-fast-")
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/t2a_fast_fleet.json"))
    args = parser.parse_args()
    shards = args.shards
    indices = (
        [int(value) for value in args.shard_indices.split(",")]
        if args.shard_indices
        else list(range(shards))
    )
    if any(index < 0 or index >= shards for index in indices):
        parser.error("--shard-indices values must be within [0, shards)")
    key = os.environ["RUNPOD_API_KEY"].strip()
    token = os.environ["HF_TOKEN"].strip()
    root = Path(__file__).resolve().parent
    script_b64 = base64.b64encode((root / "transcribe_audios2.py").read_bytes()).decode()
    creators_b64 = base64.b64encode(
        (root / "transcribe_shards" / "tinkerspace.txt").read_bytes()
    ).decode()
    public_key = (Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()

    created: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                create_one,
                key,
                token,
                public_key,
                script_b64,
                creators_b64,
                shards,
                index,
                args.include_creators,
                args.pod_prefix,
            ): index
            for index in indices
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                pod = future.result()
            except Exception as error:
                print(f"FAIL shard={index}: {error}", flush=True)
            else:
                created.append(pod)
                print(
                    f"CREATED shard={index} id={pod['id']} "
                    f"${pod.get('costPerHr')}/h",
                    flush=True,
                )

    state = {
        "created_at": time.time(),
        "requested_shards": shards,
        "pods": created,
    }
    args.state_file.write_text(json.dumps(state, indent=2))
    print(f"FLEET_CREATED {len(created)}/{len(indices)} requested shards", flush=True)
    return 0 if len(created) == len(indices) else 1


if __name__ == "__main__":
    raise SystemExit(main())
