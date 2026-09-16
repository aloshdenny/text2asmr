#!/usr/bin/env python3
"""Launch a parallel RunPod fleet for CLAP-labeling aoxo/audios2."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import textwrap
import time
import urllib.request
from pathlib import Path

GQL = "https://api.runpod.io/graphql"
GPU_ID = "NVIDIA GeForce RTX 4090"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CLOUD_TYPE = "SECURE"  # community had CUDA-passthrough failures; secure 4090 is the proven path


def gql(query: str, key: str) -> dict:
    request = urllib.request.Request(
        f"{GQL}?api_key={key}",
        data=json.dumps({"query": query}).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 text2asmr-clap-fleet",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(result["errors"])
    return result["data"]


def bootstrap(script_b64: str, ontology_b64: str, segment_b64: str,
              shards: int, index: int) -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        exec > >(tee -a /workspace/bootstrap.log) 2>&1
        echo "BOOT clap shard={index}/{shards} $(date -u)"
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq ffmpeg >/dev/null
        python -m pip install -q -U pip
        python -m pip install -q "huggingface_hub[hf_xet]" transformers soundfile numpy
        mkdir -p /workspace/text2asmr/scripts /workspace/text2asmr/text2asmr/data
        echo {script_b64} | base64 -d > /workspace/text2asmr/scripts/clap_label_audios2.py
        echo {ontology_b64} | base64 -d > /workspace/text2asmr/text2asmr/data/ontology.py
        echo {segment_b64} | base64 -d > /workspace/text2asmr/text2asmr/data/segment.py
        printf '' > /workspace/text2asmr/text2asmr/__init__.py
        printf '' > /workspace/text2asmr/text2asmr/data/__init__.py
        export PYTHONPATH=/workspace/text2asmr
        export CLAP_LABEL_BASE=/workspace/clap_audios2
        export PYTHONUNBUFFERED=1
        export HF_XET_HIGH_PERFORMANCE=1
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        python /workspace/text2asmr/scripts/clap_label_audios2.py \\
          --num-shards {shards} --shard-index {index} \\
          --batch 32 --download-workers 8 --push-every 48
        echo "CLAP_DONE shard={index}"
        """
    ).strip()


def create_one(key, token, public_key, script_b64, ontology_b64, segment_b64,
               shards, index, pod_prefix, gpu_id=None) -> dict:
    gpu_id = gpu_id or GPU_ID
    env = {
        "HF_TOKEN": token,
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PUBLIC_KEY": public_key,
        "TEXT2ASMR_BOOTSTRAP": bootstrap(
            script_b64, ontology_b64, segment_b64, shards, index
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
        containerDiskInGb: 80
        minVcpuCount: 8
        minMemoryInGb: 32
        gpuTypeId: {json.dumps(gpu_id)}
        name: {json.dumps(f"{pod_prefix}{index:02d}")}
        imageName: {json.dumps(IMAGE)}
        dockerArgs: ""
        ports: "22/tcp"
        volumeMountPath: "/workspace"
        startSsh: true
        env: [{env_gql}]
      }}) {{ id name costPerHr machineId }}
    }}
    """
    return gql(query, key)["podFindAndDeployOnDemand"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="?", type=int, default=12)
    parser.add_argument("--pod-prefix", default="t2a-clap-")
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/t2a_clap_fleet.json"))
    args = parser.parse_args()

    key = os.environ["RUNPOD_API_KEY"].strip()
    token = os.environ["HF_TOKEN"].strip()
    root = Path(__file__).resolve().parent
    data = root.parent / "text2asmr" / "data"
    script_b64 = base64.b64encode((root / "clap_label_audios2.py").read_bytes()).decode()
    ontology_b64 = base64.b64encode((data / "ontology.py").read_bytes()).decode()
    segment_b64 = base64.b64encode((data / "segment.py").read_bytes()).decode()
    public_key = (Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()

    created = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {
            ex.submit(
                create_one, key, token, public_key, script_b64, ontology_b64,
                segment_b64, args.shards, i, args.pod_prefix,
            ): i
            for i in range(args.shards)
        }
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                pod = fut.result()
            except Exception as err:
                print(f"FAIL shard={i}: {err}", flush=True)
            else:
                if pod is None:
                    print(f"FAIL shard={i}: null pod (no capacity?)", flush=True)
                    continue
                created.append(pod)
                print(
                    f"CREATED shard={i} id={pod['id']} ${pod.get('costPerHr')}/h",
                    flush=True,
                )

    args.state_file.write_text(json.dumps({
        "created_at": time.time(),
        "shards": args.shards,
        "pods": created,
        "prefix": args.pod_prefix,
    }, indent=2))
    burn = sum(float(p.get("costPerHr") or 0) for p in created)
    print(f"FLEET_CREATED {len(created)}/{args.shards} burn=${burn:.2f}/h", flush=True)
    print("ETA best ~45-75min wall if all 12 land; worst ~2-3h. Cap spend ~$5.", flush=True)
    return 0 if created else 1


if __name__ == "__main__":
    raise SystemExit(main())
