#!/usr/bin/env python3
"""Launch a high-bandwidth CPU pod that resumes Gemini labelling on crash/restart.

Hub REST is capped at 1000 req / 5 min per token; the pod saturates that plus
Xet high-performance transfers. The container start command waits for
/workspace/READY then runs scripts/supervise_gemini_label.sh forever.

    source ~/.t2a_env
    python3 scripts/launch_gemini_cpu_pod.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REST = "https://rest.runpod.io/v1/pods"
GQL = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME = "t2a-gemini-cpu"
STATE = Path("/tmp/t2a_gemini_cpu.json")
SSH_QUERY = (
    "query { myself { pods { id desiredStatus runtime { "
    "uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } } }"
)


def load_env() -> None:
    p = Path.home() / ".t2a_env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[7:]
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip().strip("'").strip('"')
        os.environ.setdefault(k, v)


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    return key


def rest(method: str, url: str, body: dict | None = None, timeout: int = 90) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 text2asmr-label-cpu",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        err = exc.read()[:800].decode()
        raise RuntimeError(f"REST {method} {url} HTTP {exc.code}: {err}") from exc


def create_pod(pubkey: str, hf_token: str) -> dict:
    env = {
        "PUBLIC_KEY": pubkey,
        "HF_TOKEN": hf_token,
        "HUGGING_FACE_HUB_TOKEN": hf_token,
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CACHE": "/workspace/hf-xet-cache",
        "HF_MAX_REQ_PER_5MIN": "980",
        "GEMINI_HUB_SEM": "64",
        "GEMINI_MAX_INFLIGHT": "80",
        "GEMINI_BACKEND": "vertex",
        "GEMINI_VERTEX_PROJECT": os.environ.get(
            "GEMINI_VERTEX_PROJECT", "project-9da5a2fe-3df4-485e-9a9"
        ),
        "GEMINI_VERTEX_LOCATION": os.environ.get("GEMINI_VERTEX_LOCATION", "global"),
        "GEMINI_BATCH_GCS": os.environ.get(
            "GEMINI_BATCH_GCS", "gs://aleddo-splitter-eval/t2a-gemini-batch"
        ),
        "GOOGLE_APPLICATION_CREDENTIALS": "/workspace/adc.json",
        "HUB_SYNC_INTERVAL": "90",
        "T2A_ROOT": "/workspace/text2asmr",
        "T2A_PYTHON": "/workspace/venv/bin/python",
        "TRANSCRIBE_BASE": "/workspace/t2a",
    }
    if os.environ.get("GEMINI_API_KEY"):
        env["GEMINI_API_KEY"] = os.environ["GEMINI_API_KEY"]
    extra = (os.environ.get("HF_TOKENS") or os.environ.get("HF_TOKEN_B") or "").strip()
    if extra:
        env["HF_TOKENS"] = extra if "," in extra or extra != hf_token else f"{hf_token},{extra}"

    attempts = [
        {
            "cloudType": "SECURE",
            "cpuFlavorIds": ["cpu5m", "cpu5g", "cpu5c"],
            "vcpuCount": 16,
            "minDownloadMbps": 8000,
            "minUploadMbps": 8000,
        },
        {
            "cloudType": "SECURE",
            "cpuFlavorIds": ["cpu5m", "cpu5g", "cpu5c", "cpu3m"],
            "vcpuCount": 16,
            "minDownloadMbps": 4000,
            "minUploadMbps": 4000,
        },
        {
            "cloudType": "COMMUNITY",
            "cpuFlavorIds": ["cpu5m", "cpu5g", "cpu5c", "cpu3m", "cpu3g", "cpu3c"],
            "vcpuCount": 16,
            "minDownloadMbps": 4000,
            "minUploadMbps": 4000,
            "supportPublicIp": True,
        },
        {
            "cloudType": "COMMUNITY",
            "cpuFlavorIds": ["cpu5m", "cpu5g", "cpu5c", "cpu3m", "cpu3g", "cpu3c"],
            "vcpuCount": 8,
            "minDownloadMbps": 2000,
            "minUploadMbps": 2000,
            "supportPublicIp": True,
        },
        {
            "cloudType": "COMMUNITY",
            "cpuFlavorIds": ["cpu5m", "cpu5g", "cpu5c", "cpu3m", "cpu3g", "cpu3c"],
            "vcpuCount": 8,
            "supportPublicIp": True,
        },
    ]
    last: Exception | None = None
    for spec in attempts:
        body = {
            "name": POD_NAME,
            "imageName": IMAGE,
            "computeType": "CPU",
            "containerDiskInGb": 80,
            "volumeInGb": 80,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
            "env": env,
            "interruptible": False,
            "cpuFlavorPriority": "availability",
            **spec,
        }
        print(
            f"create try cloud={spec.get('cloudType')} vcpu={spec.get('vcpuCount')} "
            f"down={spec.get('minDownloadMbps', 0)}",
            flush=True,
        )
        try:
            return rest("POST", REST, body)
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  failed: {exc}", flush=True)
    raise RuntimeError(f"could not create CPU pod: {last}")


def pod_id_of(created: dict) -> str:
    for key in ("id", "podId"):
        if created.get(key):
            return str(created[key])
    inner = created.get("pod") or created.get("data") or {}
    if isinstance(inner, dict) and inner.get("id"):
        return str(inner["id"])
    raise RuntimeError(f"no pod id in create response keys={list(created)[:20]}")


def gql(query: str) -> dict:
    req = urllib.request.Request(
        f"{GQL}?api_key={api_key()}",
        data=json.dumps({"query": query}).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 text2asmr-label-cpu",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        out = json.loads(resp.read())
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]


def wait_ssh(pod_id: str, timeout: int = 900) -> tuple[str, int]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = gql(SSH_QUERY)
        for pod in (data.get("myself") or {}).get("pods") or []:
            if pod.get("id") != pod_id:
                continue
            runtime = pod.get("runtime") or {}
            for port in runtime.get("ports") or []:
                if not isinstance(port, dict):
                    continue
                priv = int(port.get("privatePort") or 0)
                pub = int(port.get("publicPort") or 0)
                ip = port.get("ip")
                if (
                    priv == 22
                    and pub
                    and ip
                    and port.get("isIpPublic")
                ):
                    print("SSH port mapped; probing sshd...", flush=True)
                    for _ in range(18):
                        z = subprocess.run(
                            [
                                "ssh", "-p", str(pub),
                                "-o", "StrictHostKeyChecking=no",
                                "-o", "UserKnownHostsFile=/dev/null",
                                "-o", "ConnectTimeout=8",
                                "-o", "BatchMode=yes",
                                "-i", str(Path.home() / ".ssh" / "id_ed25519"),
                                f"root@{ip}",
                                "echo SSH_OK",
                            ],
                            capture_output=True, text=True, timeout=20,
                        )
                        if z.returncode == 0 and "SSH_OK" in (z.stdout or ""):
                            return str(ip), pub
                        err = (z.stderr or "").replace("\n", " ")[-120:]
                        print(f"  sshd not ready rc={z.returncode} {err}", flush=True)
                        time.sleep(8)
            print(f"waiting for SSH... status={pod.get('desiredStatus')} runtime={bool(runtime)}", flush=True)
        time.sleep(8)
    raise TimeoutError("SSH port not ready")


def ssh(host: str, port: int, remote: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-p",
            str(port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=25",
            "-o",
            "ServerAliveInterval=15",
            "-i",
            str(Path.home() / ".ssh" / "id_ed25519"),
            f"root@{host}",
            remote,
        ],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def scp(host: str, port: int, local: str, remote: str, timeout: int = 1800) -> None:
    z = subprocess.run(
        [
            "scp",
            "-P",
            str(port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=25",
            "-i",
            str(Path.home() / ".ssh" / "id_ed25519"),
            local,
            f"root@{host}:{remote}",
        ],
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if z.returncode != 0:
        raise RuntimeError(f"scp {local} failed: {(z.stderr or '')[-400:]}")


def pack_ledgers(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lt = HERE / "label_tool"
    with tarfile.open(dest, "w:gz") as tar:
        for pat in (
            "audios2_file_list.json",
            "gemini_audios2*.jsonl",
            "clap_train_audios2*.jsonl",
            "gemini_batch_jobs*.jsonl",
        ):
            for p in sorted(lt.glob(pat)):
                if p.is_file() and p.stat().st_size > 0:
                    tar.add(p, arcname=p.name)
    return dest


def kick(host: str, port: int) -> None:
    print("packing ledgers...", flush=True)
    tarball = Path("/tmp/t2a_gemini_ledgers.tar.gz")
    pack_ledgers(tarball)
    print(f"ledger archive {tarball.stat().st_size / 1e6:.1f} MB", flush=True)

    remote_prep = """
set -e
export DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/text2asmr/scripts /workspace/text2asmr/text2asmr/data \
         /workspace/text2asmr/label_tool /workspace/t2a /workspace/hf-xet-cache
apt-get update -qq
apt-get install -y -qq git ffmpeg python3 python3-venv python3-pip >/dev/null
python3 -m venv /workspace/venv
/workspace/venv/bin/pip install -q -U pip
/workspace/venv/bin/pip install -q "huggingface_hub[hf_xet]" google-genai \
    google-cloud-storage google-auth numpy
printf '' > /workspace/text2asmr/text2asmr/__init__.py
printf '' > /workspace/text2asmr/text2asmr/data/__init__.py
printf '' > /workspace/text2asmr/scripts/__init__.py
echo PREP_OK
"""
    z = ssh(host, port, remote_prep, timeout=600)
    print((z.stdout or "")[-1500:], flush=True)
    if z.stderr:
        print((z.stderr or "")[-800:], flush=True)
    if "PREP_OK" not in (z.stdout or ""):
        raise RuntimeError(f"prep failed rc={z.returncode} {(z.stderr or '')[-600:]}")

    files = [
        (HERE / "scripts/annotate_audios2_gemini.py", "/workspace/text2asmr/scripts/annotate_audios2_gemini.py"),
        (HERE / "scripts/annotate_audios2_gemini_batch.py", "/workspace/text2asmr/scripts/annotate_audios2_gemini_batch.py"),
        (HERE / "scripts/retag_triggers_gemini.py", "/workspace/text2asmr/scripts/retag_triggers_gemini.py"),
        (HERE / "scripts/build_clap_finetune_manifest.py", "/workspace/text2asmr/scripts/build_clap_finetune_manifest.py"),
        (HERE / "scripts/sync_clap_ft_manifest.sh", "/workspace/text2asmr/scripts/sync_clap_ft_manifest.sh"),
        (HERE / "scripts/supervise_gemini_label.sh", "/workspace/text2asmr/scripts/supervise_gemini_label.sh"),
        (HERE / "text2asmr/data/ontology.py", "/workspace/text2asmr/text2asmr/data/ontology.py"),
        (HERE / "text2asmr/data/segment.py", "/workspace/text2asmr/text2asmr/data/segment.py"),
        (tarball, "/workspace/ledgers.tar.gz"),
        (Path.home() / ".config/gcloud/application_default_credentials.json", "/workspace/adc.json"),
    ]
    for local, remote in files:
        print(f"scp {local.name}...", flush=True)
        scp(host, port, str(local), remote)

    z = ssh(
        host,
        port,
        "set -e; tar -xzf /workspace/ledgers.tar.gz -C /workspace/text2asmr/label_tool; "
        "chmod +x /workspace/text2asmr/scripts/supervise_gemini_label.sh "
        "/workspace/text2asmr/scripts/sync_clap_ft_manifest.sh; "
        "cat >/etc/cron.d/t2a-gemini <<'EOF'\n"
        "* * * * * root test -f /workspace/READY && "
        "! pgrep -f supervise_gemini_label.sh >/dev/null && "
        "nohup bash /workspace/text2asmr/scripts/supervise_gemini_label.sh "
        ">> /workspace/supervise.log 2>&1 &\n"
        "EOF\n"
        "chmod 644 /etc/cron.d/t2a-gemini; "
        "service cron start >/dev/null 2>&1 || cron >/dev/null 2>&1 || true; "
        "touch /workspace/READY; sleep 8; "
        "if ! pgrep -f supervise_gemini_label.sh >/dev/null; then "
        "  nohup bash /workspace/text2asmr/scripts/supervise_gemini_label.sh "
        "    >> /workspace/supervise.log 2>&1 & echo FALLBACK_SUP $!; "
        "fi; "
        "pgrep -af supervise_gemini || true; "
        "pgrep -af annotate_audios2_gemini_batch || true; "
        "echo KICK_DONE",
        timeout=300,
    )
    print((z.stdout or "")[-2000:], flush=True)
    print((z.stderr or "")[-800:], flush=True)
    if "KICK_DONE" not in (z.stdout or ""):
        raise RuntimeError(f"kick failed rc={z.returncode}")


def main() -> int:
    load_env()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-id", default="", help="kick an already-created pod")
    args = ap.parse_args()

    hf = os.environ.get("HF_TOKEN", "").strip()
    if not hf:
        print("HF_TOKEN is not set", flush=True)
        return 2

    if args.pod_id:
        pid = args.pod_id
        cost = "?"
        print(f"resuming kick on pod {pid}", flush=True)
    else:
        pubkey = (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
        created = create_pod(pubkey, hf)
        print(json.dumps({k: created.get(k) for k in created if k != "env"}, indent=2)[:2000], flush=True)
        pid = pod_id_of(created)
        cost = created.get("costPerHr") or created.get("costPerHour") or "?"
        print(f"pod {pid} costPerHr={cost}", flush=True)

    print(
        "ETA: best ~2–4d (Vertex batch + Hub saturated), worst ~8–12d "
        "(Hub 429 / Vertex quota). Recalc after first harvest line.",
        flush=True,
    )

    host, port = wait_ssh(pid)
    print(f"SSH {host}:{port}", flush=True)
    kick(host, port)

    STATE.write_text(json.dumps({"pod": pid, "ssh": {"host": host, "port": port}, "costPerHr": cost}, indent=2))
    print(f"state {STATE}", flush=True)
    print(
        f"arm watcher: python3 -u scripts/watch_gemini_cpu_pod.py "
        f"--pod-id {pid} --ssh-host {host} --ssh-port {port}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
