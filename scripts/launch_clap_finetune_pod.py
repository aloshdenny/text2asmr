#!/usr/bin/env python3
"""Launch one RunPod 4090, wait for SSH, kick CLAP fine-tune bootstrap."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

GQL = "https://api.runpod.io/graphql"
GPU_ID = "NVIDIA GeForce RTX 4090"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
HERE = Path(__file__).resolve().parents[1]
MANIFEST_REPO = "aoxo/clap-ft-data"
MANIFEST_FILE = "clap_finetune_manifest.jsonl"
PUSH_DEFAULT = "aoxo/clap-htsat-fused-asmr"


def gql(query: str, key: str) -> dict:
    request = urllib.request.Request(
        f"{GQL}?api_key={key}",
        data=json.dumps({"query": query}).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 text2asmr-clap-ft",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(result["errors"])
    return result["data"]


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def create_pod(key: str, token: str, public_key: str) -> dict:
    env = {
        "HF_TOKEN": token,
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PUBLIC_KEY": public_key,
    }
    env_gql = ", ".join(
        f'{{key:{json.dumps(k)},value:{json.dumps(v)}}}' for k, v in env.items()
    )
    mutation = f"""
    mutation {{
      podFindAndDeployOnDemand(input: {{
        cloudType: COMMUNITY
        gpuCount: 1
        volumeInGb: 50
        containerDiskInGb: 80
        minVcpuCount: 8
        minMemoryInGb: 32
        gpuTypeId: "{GPU_ID}"
        name: "t2a-clap-ft"
        imageName: "{IMAGE}"
        dockerArgs: ""
        ports: "22/tcp"
        volumeMountPath: "/workspace"
        env: [{env_gql}]
        startSsh: true
      }}) {{
        id
        name
        desiredStatus
        costPerHr
        machine {{ podHostId }}
      }}
    }}
    """
    return gql(mutation, key)["podFindAndDeployOnDemand"]


def wait_ssh(key: str, pod_id: str, timeout: int = 600) -> tuple[str, int]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = gql(
            "query { myself { pods { id runtime { ports { ip isIpPublic privatePort publicPort type } } } } }",
            key,
        )
        for p in data["myself"]["pods"]:
            if p["id"] != pod_id:
                continue
            for port in (p.get("runtime") or {}).get("ports") or []:
                if (
                    port.get("type") == "tcp"
                    and int(port.get("privatePort") or 0) == 22
                    and port.get("isIpPublic")
                ):
                    print("SSH port up; settling 20s for GPU...", flush=True)
                    time.sleep(20)
                    return port["ip"], int(port["publicPort"])
        print("waiting for SSH...", flush=True)
        time.sleep(10)
    raise TimeoutError("SSH port not ready")


def kick(host: str, port: int, push_to: str, hf_token: str) -> None:
    script_b64 = b64(HERE / "scripts/finetune_clap.py")
    ont_b64 = b64(HERE / "text2asmr/data/ontology.py")
    tok = hf_token.replace("'", "'\"'\"'")
    remote = f"""
set -e
export HF_TOKEN='{tok}'
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES=0
if pgrep -f '[f]inetune_clap' >/dev/null; then echo ALREADY; exit 0; fi
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ffmpeg >/dev/null
python -m pip install -q -U pip
python -m pip install -q "huggingface_hub[hf_xet]" "transformers==4.44.2" soundfile "numpy<2" accelerate
# Wait until torch can see the GPU (driver attach race on some hosts)
for i in $(seq 1 60); do
  if python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'; then
    echo CUDA_READY
    break
  fi
  echo "cuda_wait $i"
  sleep 5
done
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
mkdir -p /workspace/text2asmr/scripts /workspace/text2asmr/text2asmr/data /workspace/text2asmr/label_tool
echo {script_b64} | base64 -d > /workspace/text2asmr/scripts/finetune_clap.py
echo {ont_b64} | base64 -d > /workspace/text2asmr/text2asmr/data/ontology.py
printf '' > /workspace/text2asmr/text2asmr/__init__.py
printf '' > /workspace/text2asmr/text2asmr/data/__init__.py
printf '' > /workspace/text2asmr/scripts/__init__.py
export PYTHONPATH=/workspace/text2asmr
export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
cd /workspace/text2asmr
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download("{MANIFEST_REPO}", "{MANIFEST_FILE}", repo_type="dataset", token=os.environ["HF_TOKEN"])
shutil.copy(p, "label_tool/{MANIFEST_FILE}")
print("manifest_lines", open("label_tool/{MANIFEST_FILE}").read().count(chr(10)))
PY
nohup python scripts/finetune_clap.py \\
  --manifest label_tool/{MANIFEST_FILE} \\
  --out /workspace/clap-ft \\
  --epochs 3 --batch 16 --lr 1e-5 \\
  --push-to {push_to} \\
  > /workspace/clap_ft.log 2>&1 &
echo KICKED_PID=$!
sleep 5
pgrep -af finetune_clap || true
echo KICK_DONE
"""
    z = subprocess.run(
        [
            "ssh", "-p", str(port),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-i", str(Path.home() / ".ssh/id_ed25519"),
            f"root@{host}",
            "bash", "-s",
        ],
        input=remote,
        text=True,
        capture_output=True,
        timeout=900,
    )
    print(z.stdout[-3000:] if z.stdout else "")
    print(z.stderr[-1500:] if z.stderr else "")
    if "KICK_DONE" not in (z.stdout or "") and "ALREADY" not in (z.stdout or ""):
        raise RuntimeError(f"kick failed rc={z.returncode}")


def main() -> int:
    key = os.environ["RUNPOD_API_KEY"]
    token = os.environ["HF_TOKEN"]
    push_to = os.environ.get("CLAP_FT_REPO", PUSH_DEFAULT)
    public_key = Path.home().joinpath(".ssh/id_ed25519.pub").read_text().strip()

    pod = create_pod(key, token, public_key)
    print(json.dumps(pod, indent=2), flush=True)
    print("ETA: best ~45–90 min, worst ~3h | ~$0.34–0.5/h typical 4090 COMMUNITY", flush=True)

    host, port = wait_ssh(key, pod["id"])
    print(f"SSH {host}:{port}", flush=True)
    kick(host, port, push_to, token)

    Path("/tmp/t2a_clap_ft.json").write_text(
        json.dumps(
            {"pod": pod, "ssh": {"host": host, "port": port}, "push_to": push_to},
            indent=2,
        )
    )
    print("kicked — arm: python3 scripts/watch_clap_finetune_pod.py --pod-id", pod["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
