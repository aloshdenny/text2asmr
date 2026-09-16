#!/usr/bin/env python3
"""Launch one secure RunPod GPU, bootstrap, run prep_clap_v2 + train_clap_v2 concurrently."""
from __future__ import annotations
import base64, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

GQL = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
HERE = Path(__file__).resolve().parents[1]
PREFS = ["RTX 6000 Ada", "L40S", "RTX 4090"]
PUSH_TO = "aoxo/clap-htsat-unfused-asmr-v2"

def gql(query, key):
    req = urllib.request.Request(f"{GQL}?api_key={key}", data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 t2a-clap-v2"})
    with urllib.request.urlopen(req, timeout=90) as r: res = json.load(r)
    if res.get("errors"): raise RuntimeError(res["errors"])
    return res["data"]

def b64(p: Path) -> str: return base64.b64encode(p.read_bytes()).decode()

def gpu_id(key, name):
    for g in gql("query { gpuTypes { id displayName } }", key)["gpuTypes"]:
        if g["displayName"] == name: return g["id"]
    raise KeyError(name)

def create(key, token, pub, gid, name):
    env = ", ".join(f'{{key:{json.dumps(k)},value:{json.dumps(v)}}}' for k, v in {"HF_TOKEN": token, "PUBLIC_KEY": pub}.items())
    m = f'''mutation {{ podFindAndDeployOnDemand(input: {{ cloudType: SECURE gpuCount: 1 volumeInGb: 0 containerDiskInGb: 220
      minVcpuCount: 12 minMemoryInGb: 48 gpuTypeId: "{gid}" name: "t2a-clap-v2" imageName: "{IMAGE}" dockerArgs: "" ports: "22/tcp"
      volumeMountPath: "/workspace" env: [{env}] startSsh: true }}) {{ id name costPerHr machine {{ podHostId }} }} }}'''
    return gql(m, key)["podFindAndDeployOnDemand"]

def wait_ssh(key, pod_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for p in gql("query { myself { pods { id runtime { ports { ip isIpPublic privatePort publicPort type } } } } }", key)["myself"]["pods"]:
            if p["id"] != pod_id: continue
            for port in (p.get("runtime") or {}).get("ports") or []:
                if port.get("type") == "tcp" and int(port.get("privatePort") or 0) == 22 and port.get("isIpPublic"):
                    time.sleep(15); return port["ip"], int(port["publicPort"])
        print("waiting for SSH...", flush=True); time.sleep(10)
    raise TimeoutError("ssh")

def kick(host, port, token):
    tok = token.replace("'", "'\"'\"'")
    remote = f"""
set -e
export HF_TOKEN='{tok}'; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1; export DEBIAN_FRONTEND=noninteractive
if pgrep -f '[p]rep_clap_v2' >/dev/null; then echo ALREADY; exit 0; fi
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
python -m pip install -q -U pip
python -m pip install -q "transformers==4.46.3" "huggingface_hub>=0.25" "numpy<2" accelerate
for i in $(seq 1 60); do python -c 'import torch; assert torch.cuda.is_available()' && break; echo cuda_wait $i; sleep 5; done
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /workspace/t2a/scripts /workspace/t2a/label_tool /workspace/mel
echo {b64(HERE/'scripts/prep_clap_v2.py')} | base64 -d > /workspace/t2a/scripts/prep_clap_v2.py
echo {b64(HERE/'scripts/train_clap_v2.py')} | base64 -d > /workspace/t2a/scripts/train_clap_v2.py
cd /workspace/t2a
python - <<'PY'
from huggingface_hub import hf_hub_download; import shutil, os
p = hf_hub_download("aoxo/clap-ft-data", "v2/train_subset.jsonl", repo_type="dataset", token=os.environ["HF_TOKEN"])
shutil.copy(p, "label_tool/train_subset.jsonl"); print("subset_lines", sum(1 for _ in open("label_tool/train_subset.jsonl")))
PY
NW=$(nproc); echo vcpu=$NW
nohup python scripts/prep_clap_v2.py --subset label_tool/train_subset.jsonl --out /workspace/mel --workers $((NW+4)) > /workspace/prep.log 2>&1 &
nohup python scripts/train_clap_v2.py --data /workspace/mel --out /workspace/clap-v2 --batch 384 --epochs 5 \\
  --wait-for-prep --min-rows 120000 --total-rows 695000 --push-to {PUSH_TO} > /workspace/train.log 2>&1 &
sleep 3; pgrep -af 'clap_v2' || true
echo KICK_DONE
"""
    z = subprocess.run(["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "ConnectTimeout=30", "-i", str(Path.home() / ".ssh/id_ed25519"), f"root@{host}", "bash", "-s"],
                       input=remote, text=True, capture_output=True, timeout=1500)
    print(z.stdout[-3000:]); print(z.stderr[-1500:])
    if "KICK_DONE" not in z.stdout and "ALREADY" not in z.stdout: raise RuntimeError(f"kick failed rc={z.returncode}")

def main():
    key = os.environ["RUNPOD_API_KEY"]; token = os.environ["HF_TOKEN"]
    pub = Path.home().joinpath(".ssh/id_ed25519.pub").read_text().strip()
    pod = None
    for name in PREFS:
        try:
            pod = create(key, token, pub, gpu_id(key, name), name); print("created", name, json.dumps(pod)); break
        except Exception as e: print(f"{name}: {str(e)[:200]}")
    if not pod: raise SystemExit("no GPU available")
    host, port = wait_ssh(key, pod["id"]); print(f"SSH {host}:{port}", flush=True)
    kick(host, port, token)
    Path("/tmp/t2a_clap_v2.json").write_text(json.dumps({"pod": pod, "ssh": {"host": host, "port": port}}, indent=2))
    print("KICKED", pod["id"])

if __name__ == "__main__": main()
