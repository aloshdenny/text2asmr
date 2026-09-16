#!/usr/bin/env python3
"""Cheapest RunPod GPU: Gemini-Pro pilot relabel + probe, plus GCS raw-predictions archive to HF."""
import base64, json, os, subprocess, time, urllib.request
from pathlib import Path
GQL = "https://api.runpod.io/graphql"; HERE = Path(__file__).resolve().parents[1]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
PREFS = ["RTX 3070", "RTX A4000", "RTX 3080 Ti", "RTX 4000 Ada SFF", "RTX A4500", "RTX 3090", "RTX 4090"]
def gql(q, key):
    req = urllib.request.Request(f"{GQL}?api_key={key}", data=json.dumps({"query": q}).encode(), headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 t2a"})
    with urllib.request.urlopen(req, timeout=90) as r: res = json.load(r)
    if res.get("errors"): raise RuntimeError(res["errors"])
    return res["data"]
def b64(p): return base64.b64encode(Path(p).read_bytes()).decode()
def main():
    key = os.environ["RUNPOD_API_KEY"]; hf = os.environ["HF_TOKEN"]
    if os.environ.get("REUSE_POD") and Path("/tmp/t2a_pilot.json").exists():
        j = json.loads(Path("/tmp/t2a_pilot.json").read_text()); return kick(j["host"], j["port"])
    gcs = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True, check=True).stdout.strip()
    pub = Path.home().joinpath(".ssh/id_ed25519.pub").read_text().strip()
    ids = {g["displayName"]: g["id"] for g in gql("query { gpuTypes { id displayName } }", key)["gpuTypes"]}
    env = ", ".join(f'{{key:{json.dumps(k)},value:{json.dumps(v)}}}' for k, v in {"HF_TOKEN": hf, "GCS_TOKEN": gcs, "PUBLIC_KEY": pub}.items())
    pod = None
    for cloud in ("COMMUNITY", "SECURE"):
        for name in PREFS:
            try:
                pod = gql(f'''mutation {{ podFindAndDeployOnDemand(input: {{ cloudType: {cloud} gpuCount: 1 volumeInGb: 0 containerDiskInGb: 60 minVcpuCount: 4 minMemoryInGb: 15
                  gpuTypeId: "{ids[name]}" name: "t2a-pilot" imageName: "{IMAGE}" dockerArgs: "" ports: "22/tcp" volumeMountPath: "/workspace" env: [{env}] startSsh: true }}) {{ id costPerHr }} }}''', key)["podFindAndDeployOnDemand"]
                print("created", cloud, name, pod, flush=True); break
            except Exception as e: print(f"{cloud} {name}: {str(e)[:80]}")
        if pod: break
    if not pod: raise SystemExit("no GPU")
    host = port = None; t0 = time.time()
    while not host and time.time() - t0 < 900:
        for p in gql("query { myself { pods { id runtime { ports { ip isIpPublic privatePort publicPort type } } } } }", key)["myself"]["pods"]:
            if p["id"] == pod["id"]:
                for pt in (p.get("runtime") or {}).get("ports") or []:
                    if pt.get("type") == "tcp" and int(pt.get("privatePort") or 0) == 22 and pt.get("isIpPublic"): host, port = pt["ip"], int(pt["publicPort"])
        if not host: time.sleep(10)
    print("SSH", host, port, flush=True); time.sleep(15)
    Path("/tmp/t2a_pilot.json").write_text(json.dumps({"id": pod["id"], "host": host, "port": port}))
    kick(host, port)

def kick(host, port):
    remote = f"""
set -e
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
export $(tr "\\0" "\\n" < /proc/1/environ | grep -E "^(HF_TOKEN|GCS_TOKEN)=" | xargs)
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
python -m pip install -q "transformers==4.46.3" "huggingface_hub>=0.25" "numpy<2"
mkdir -p /workspace/t2a/scripts /workspace/t2a/label_tool
echo {b64(HERE/'scripts/pilot_gemini_pro.py')} | base64 -d > /workspace/t2a/scripts/pilot_gemini_pro.py
echo {b64(HERE/'scripts/pilot_probe.py')} | base64 -d > /workspace/t2a/scripts/pilot_probe.py
echo {b64('/private/tmp/claude-501/-Users-aoxo-vscode/3b5b2ee2-3615-47d6-9e67-bd48873d7b13/scratchpad/archive_preds.py')} | base64 -d > /workspace/archive_preds.py
cd /workspace/t2a
python - <<'PY'
from huggingface_hub import hf_hub_download; import shutil, os
p = hf_hub_download("aoxo/clap-ft-data", "v2/train_subset_balanced.jsonl", repo_type="dataset", token=os.environ["HF_TOKEN"]); shutil.copy(p, "label_tool/subset.jsonl")
PY
nohup bash -c 'python scripts/pilot_gemini_pro.py --subset label_tool/subset.jsonl --out /workspace/pilot --per-class 120 --workers 12 && python scripts/pilot_probe.py /workspace/pilot' > /workspace/pilot.log 2>&1 &
cd /workspace && sed -i 's#subprocess.run(\\["gcloud","auth","print-access-token"\\],capture_output=True,text=True,check=True).stdout.strip()#os.environ["GCS_TOKEN"]#' archive_preds.py
nohup bash -c 'python archive_preds.py && tar -czf vertex_predictions_raw.tar.gz vertex_preds && python -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=\\"vertex_predictions_raw.tar.gz\\", path_in_repo=\\"raw/vertex_predictions_raw.tar.gz\\", repo_id=\\"aoxo/clap-ft-data\\", repo_type=\\"dataset\\", commit_message=\\"archive raw Vertex Gemini batch predictions (10760 jobs)\\"); print(\\"ARCHIVE_UPLOAD_OK\\")"' > /workspace/archive.log 2>&1 &
sleep 3; pgrep -af "pilot_gemini|archive_preds" | grep -v pgrep | cut -c1-80
echo KICK_DONE
"""
    z = subprocess.run(["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=30",
                        "-i", str(Path.home() / ".ssh/id_ed25519"), f"root@{host}", "bash", "-s"], input=remote, text=True, capture_output=True, timeout=1500)
    print(z.stdout[-2500:]); print(z.stderr[-800:])
    if "KICK_DONE" not in z.stdout: raise SystemExit("kick failed")
if __name__ == "__main__": main()
