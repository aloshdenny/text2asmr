#!/usr/bin/env python3
"""RunPod GPU running one shard of transcribe_audios2.py (faster-whisper large-v3) with the speech gate."""
import json, os, subprocess, time
from pathlib import Path
import launch_clap_v2_pod as L
HERE = Path(__file__).resolve().parents[1]
def main():
    key = os.environ["RUNPOD_API_KEY"]; token = os.environ["HF_TOKEN"]; pub = Path.home().joinpath(".ssh/id_ed25519.pub").read_text().strip()
    repo = os.environ.get("T2A_REPO", "aoxo/audios3"); shards = os.environ.get("T2A_SHARDS", "2"); idx = os.environ.get("T2A_SHARD", "1"); workers = os.environ.get("T2A_WORKERS", "4")
    if os.environ.get("REUSE_POD") and Path("/tmp/t2a_transcribe_pod.json").exists():
        j = json.loads(Path("/tmp/t2a_transcribe_pod.json").read_text()); return kick(j["host"], j["port"], repo, shards, idx, workers)
    pod = None
    for name in os.environ.get("T2A_PREFS", "RTX 4090,RTX A5000,RTX A6000,L40S").split(","):
        for cloud in ("SECURE", "COMMUNITY"):
            try: pod = L.create(key, token, pub, L.gpu_id(key, name), name, cloud); print("created", cloud, name, pod, flush=True); break
            except Exception as e: print(f"{cloud} {name}: {str(e)[:80]}")
        if pod: break
    if not pod: raise SystemExit("no GPU")
    host, port = L.wait_ssh(key, pod["id"]); print("SSH", host, port, flush=True)
    Path("/tmp/t2a_transcribe_pod.json").write_text(json.dumps({"id": pod["id"], "host": host, "port": port}))
    kick(host, port, repo, shards, idx, workers)

def kick(host, port, repo, shards, idx, workers):
    remote = f"""
set -e
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
export $(tr "\\0" "\\n" < /proc/1/environ | grep -E "^HF_TOKEN=" | xargs)
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
python -m pip install -q -U faster-whisper "huggingface_hub>=0.25" "numpy<2" nvidia-cudnn-cu12 nvidia-cublas-cu12 2>&1 | grep -vE "WARNING|notice" | tail -1
export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas.lib, nvidia.cudnn.lib; print(list(nvidia.cublas.lib.__path__)[0] + ':' + list(nvidia.cudnn.lib.__path__)[0])"):$LD_LIBRARY_PATH
mkdir -p /workspace/t2a/scripts /workspace/t2a/text2asmr/data && cd /workspace/t2a
echo {L.b64(HERE/'scripts/transcribe_audios2.py')} | base64 -d > scripts/transcribe_audios2.py
echo {L.b64(HERE/'text2asmr/data/segment.py')} | base64 -d > text2asmr/data/segment.py; touch text2asmr/__init__.py text2asmr/data/__init__.py
cat > /workspace/run_transcribe.sh <<EOF2
#!/bin/bash
export \\$(tr "\\0" "\\n" < /proc/1/environ | grep -E "^HF_TOKEN=" | xargs); export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/t2a LD_LIBRARY_PATH=$LD_LIBRARY_PATH
cd /workspace/t2a
while true; do python scripts/transcribe_audios2.py --repo {repo} --model large-v3 --compute-type float16 --transcribe-workers {workers} --producer-workers {workers} --upload-batch-size 64 --upload-batch-timeout 1500 --num-shards {shards} --shard-index {idx}; echo "[\\$(date +%T)] exited rc=\\$?, restarting in 60s"; sleep 60; done
EOF2
chmod +x /workspace/run_transcribe.sh
setsid nohup /workspace/run_transcribe.sh > /workspace/transcribe.log 2>&1 < /dev/null &
sleep 5; pgrep -af "transcribe_audios2" | grep -v pgrep | wc -l
echo KICK_DONE
"""
    z = subprocess.run(["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=30", "-i", str(Path.home() / ".ssh/id_ed25519"), f"root@{host}", "bash", "-s"], input=remote, text=True, capture_output=True, timeout=1500)
    print(z.stdout[-1500:]); print(z.stderr[-600:])
    if "KICK_DONE" not in z.stdout: raise SystemExit("kick failed")
if __name__ == "__main__": main()
