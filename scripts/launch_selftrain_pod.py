#!/usr/bin/env python3
import base64, json, os, subprocess, time
from pathlib import Path
import launch_clap_v2_pod as L
HERE = Path(__file__).resolve().parents[1]
def kick(host, port):
    files = ["scripts/prep_clap_v2.py", "scripts/train_clap_v3.py", "scripts/pseudo_label_clap.py", "scripts/selftrain_clap.sh", "scripts/prep_yt_chapters.py", "scripts/clapv4.sh", "scripts/train_clap_v5.py", "scripts/clapv5.sh"]
    writes = "\n".join(f"echo {L.b64(HERE / f)} | base64 -d > /workspace/t2a/{f}" for f in files)
    remote = f"""
set -e
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
export $(tr "\\0" "\\n" < /proc/1/environ | grep -E "^HF_TOKEN=" | xargs)
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
python -m pip install -q "transformers==4.46.3" "huggingface_hub>=0.25" "numpy<2"
mkdir -p /workspace/t2a/scripts /workspace/t2a/label_tool
{writes}
chmod +x /workspace/t2a/scripts/*.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc
setsid nohup /workspace/t2a/scripts/{os.environ.get("T2A_PIPE", "selftrain_clap.sh")} > /workspace/pipeline.log 2>&1 < /dev/null &
sleep 3; pgrep -af "clap.*\.sh" | grep -v pgrep | wc -l
echo KICK_DONE
"""
    z = subprocess.run(["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=30", "-i", str(Path.home() / ".ssh/id_ed25519"), f"root@{host}", "bash", "-s"], input=remote, text=True, capture_output=True, timeout=1500)
    print(z.stdout[-1500:]); print(z.stderr[-600:])
    if "KICK_DONE" not in z.stdout: raise SystemExit("kick failed")
def main():
    key = os.environ["RUNPOD_API_KEY"]; token = os.environ["HF_TOKEN"]; pub = Path.home().joinpath(".ssh/id_ed25519.pub").read_text().strip()
    if os.environ.get("REUSE_POD") and Path("/tmp/t2a_selftrain.json").exists():
        j = json.loads(Path("/tmp/t2a_selftrain.json").read_text()); return kick(j["host"], j["port"])
    pod = None
    for name in os.environ.get("T2A_PREFS", "RTX 6000 Ada,L40S,RTX A6000,L40,A40,RTX 4090,RTX A5000").split(","):
        try: pod = L.create(key, token, pub, L.gpu_id(key, name), name, "SECURE"); print("created", name, pod, flush=True); break
        except Exception as e: print(f"{name}: {str(e)[:80]}")
    if not pod: raise SystemExit("no GPU")
    host, port = L.wait_ssh(key, pod["id"]); print("SSH", host, port, flush=True)
    Path("/tmp/t2a_selftrain.json").write_text(json.dumps({"id": pod["id"], "host": host, "port": port})); kick(host, port)
if __name__ == "__main__": main()
