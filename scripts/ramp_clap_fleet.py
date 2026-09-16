#!/usr/bin/env python3
"""Ramp CLAP-label pods one-at-a-time once the newest is saturated.

Policy:
  - Launch shard 0 first with fine-tuned CLAP.
  - Only launch shard N+1 after the latest pod shows util>=90% AND
    memory util>=70% for two consecutive polls (or labeling progress).
  - Cap concurrent pods and spend; terminate on CLAP_DONE / idle / budget.

  source ~/.t2a_env
  python3 scripts/ramp_clap_fleet.py --shards 12 --max-pods 8
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import scripts.launch_clap_fleet as launch

# Ignore tty signals so nested SSH PTYs cannot kill the ramp under nohup.
for _sig in (getattr(signal, "SIGTTOU", None), getattr(signal, "SIGTTIN", None),
             getattr(signal, "SIGHUP", None)):
    if _sig is not None:
        try:
            signal.signal(_sig, signal.SIG_IGN)
        except Exception:
            pass

GQL = "https://api.runpod.io/graphql"
PREFIX = "t2a-clap-"
STATE = Path("/tmp/t2a_clap_fleet.json")
CLAP_MODEL = "aoxo/clap-htsat-unfused-asmr"
UTIL_READY = 90.0
# CLAP inference will not fill 24GB; require real compute, not training-style VRAM.
MEM_READY = 20.0
MAX_SPEND_USD = 40.0
BALANCE_FLOOR = 8.0


def gql(query: str, key: str) -> dict:
    request = urllib.request.Request(
        f"{GQL}?api_key={key}",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 t2a-clap-ramp"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(result["errors"])
    return result["data"]


def terminate(key: str, pod_id: str) -> None:
    gql(f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}', key)


def list_clap_pods(key: str) -> tuple[float, list[dict]]:
    data = gql(
        """query { myself { clientBalance pods {
            id name costPerHr desiredStatus
            machine { podHostId }
            runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } }
        } } }""",
        key,
    )["myself"]
    pods = [p for p in data.get("pods") or [] if (p.get("name") or "").startswith(PREFIX)]
    return float(data.get("clientBalance") or 0), pods


def ssh_cmd(host_id: str, cmd: str, timeout: int = 30) -> str:
    """Run a remote command via RunPod SSH (requires a PTY)."""
    import pty as _pty

    master, slave = _pty.openpty()
    try:
        proc = subprocess.Popen(
            [
                "ssh", "-tt",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=10",
                "-o", "LogLevel=ERROR",
                "-i", str(Path.home() / ".ssh" / "id_ed25519"),
                f"{host_id}@ssh.runpod.io",
            ],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave)
        slave = -1
        payload = (cmd.rstrip() + "\nexit\n").encode()
        os.write(master, payload)
        chunks: list[bytes] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                # drain remaining
                try:
                    while True:
                        import select
                        r, _, _ = select.select([master], [], [], 0.05)
                        if not r:
                            break
                        data = os.read(master, 4096)
                        if not data:
                            break
                        chunks.append(data)
                except OSError:
                    pass
                break
            import select
            r, _, _ = select.select([master], [], [], 0.5)
            if r:
                try:
                    data = os.read(master, 4096)
                except OSError:
                    break
                if not data:
                    break
                chunks.append(data)
                if b"Connection to ssh.runpod.io closed" in b"".join(chunks[-5:]):
                    # wait briefly for exit
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    break
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        return b"".join(chunks).decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError) as exc:
        return f"SSH_FAIL {type(exc).__name__}"
    finally:
        try:
            os.close(master)
        except OSError:
            pass
        if slave >= 0:
            try:
                os.close(slave)
            except OSError:
                pass


def ensure_bootstrap(host_id: str | None) -> str:
    if not host_id:
        return "NO_HOST"
    out = ssh_cmd(
        host_id,
        """
if pgrep -f '[c]lap_label_audios2' >/dev/null; then
  echo T2A_BOOT=RUNNING
elif pgrep -af '[b]ash /workspace/run.sh' >/dev/null; then
  echo T2A_BOOT=STARTING
else
  # Never replay TEXT2ASMR_BOOTSTRAP — it embeds a stale script and
  # overwrites the GPU pipeline. Restart from the on-disk kick script.
  if [ -x /tmp/kick_clap.sh ]; then
    nohup /tmp/kick_clap.sh >> /workspace/clap_label.log 2>&1 &
    echo T2A_BOOT=STARTED
  else
    echo T2A_BOOT=NEED_KICK
  fi
fi
tail -n 3 /workspace/bootstrap.log 2>/dev/null || true
""",
        timeout=35,
    )
    # Prefer live-process markers; ssh -tt echoes the script so unprefixed
    # BOOT_STARTED would false-positive.
    if "T2A_BOOT=RUNNING" in out or "T2A_BOOT=STARTING" in out:
        return "RUNNING"
    if "T2A_BOOT=STARTED" in out:
        return "STARTED"
    if "T2A_BOOT=NEED_KICK" in out:
        return "NEED_KICK"
    if "T2A_BOOT=CUDA_FAIL_SKIP" in out:
        return "CUDA_FAIL"
    return "UNKNOWN"


def pod_cuda_bad(host_id: str | None) -> bool:
    if not host_id:
        return False
    # Use unique markers so echoed SSH commands cannot false-positive.
    out = ssh_cmd(
        host_id,
        """
python - <<'PY'
import torch
print('T2A_CUDA=' + ('1' if torch.cuda.is_available() else '0'))
PY
tail -n 8 /workspace/bootstrap.log 2>/dev/null || true
""",
        timeout=45,
    )
    if "T2A_CUDA=1" in out:
        return False
    if "T2A_CUDA=0" in out:
        return True
    return bool(re.search(r"(?m)^CUDA_FAIL\b", out))


def pod_done(host_id: str | None) -> bool:
    if not host_id:
        return False
    out = ssh_cmd(
        host_id,
        "tail -n 40 /workspace/bootstrap.log 2>/dev/null || true",
        timeout=35,
    )
    # Require the real log line — not the echoed grep/command text.
    return bool(re.search(r"(?m)^CLAP_DONE shard=\d+\s*$", out))


def pod_saturated(pod: dict) -> bool:
    runtime = pod.get("runtime") or {}
    gpus = runtime.get("gpus") or [{}]
    g = gpus[0] if gpus else {}
    util = float(g.get("gpuUtilPercent") or 0)
    mem = float(g.get("memoryUtilPercent") or 0)
    up = float(runtime.get("uptimeInSeconds") or 0)
    return up > 120 and util >= UTIL_READY and mem >= MEM_READY


def bootstrap_ft(script_b64: str, ontology_b64: str, segment_b64: str,
                 shards: int, index: int, batch: int, workers: int,
                 clap_model: str) -> str:
    # Override launch.bootstrap with FT model + fatter I/O.
    return f"""
set -euo pipefail
exec > >(tee -a /workspace/bootstrap.log) 2>&1
echo "BOOT clap-ft shard={index}/{shards} model={clap_model} $(date -u)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ffmpeg >/dev/null
python -m pip install -q -U pip
# Image has torch 2.4; transformers 5.x requires torch>=2.5 and breaks ClapModel.
python -m pip install -q "huggingface_hub[hf_xet]" "transformers==4.46.3" soundfile numpy
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
export CLAP_MODEL={clap_model}
export CLAP_PROCESSOR=laion/clap-htsat-unfused
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Community hosts sometimes expose /dev/nvidiaN without /dev/nvidia0; PyTorch needs nvidia0.
if [ ! -e /dev/nvidia0 ]; then
  for d in /dev/nvidia[0-9]*; do
    case "$d" in *nvidia-*) continue ;; esac
    [ -e "$d" ] || continue
    ln -sf "$d" /dev/nvidia0
    echo "LINKED $d -> /dev/nvidia0"
    break
  done
fi
# Wait until CUDA is usable (pod sometimes reports GPU before driver is ready).
for i in 1 2 3 4 5 6 7 8 9 10; do
  if python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    echo "CUDA_READY attempt=$i"
    break
  fi
  echo "CUDA_WAIT attempt=$i"
  sleep 5
done
python -c 'import torch; print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
if ! python -c 'import torch; assert torch.cuda.is_available()'; then
  echo "CUDA_FAIL shard={index}"
  exit 42
fi
python /workspace/text2asmr/scripts/clap_label_audios2.py \\
  --num-shards {shards} --shard-index {index} \\
  --clap-model {clap_model} \\
  --batch {batch} --download-workers {workers} --push-every 128
echo "CLAP_DONE shard={index}"
""".strip()


def create_shard(key: str, token: str, pub: str, script_b64: str, ontology_b64: str,
                 segment_b64: str, shards: int, index: int, prefix: str,
                 batch: int, workers: int, clap_model: str) -> dict:
    # Monkeypatch bootstrap used by create_one
    launch.bootstrap = lambda sb, ob, segb, sh, idx: bootstrap_ft(
        sb, ob, segb, sh, idx, batch, workers, clap_model
    )
    return launch.create_one(
        key, token, pub, script_b64, ontology_b64, segment_b64, shards, index, prefix
    )


def save_state(pods: list[dict], shards: int, prefix: str, started: float) -> None:
    STATE.write_text(json.dumps({
        "created_at": started,
        "shards": shards,
        "prefix": prefix,
        "pods": pods,
        "clap_model": CLAP_MODEL,
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--max-pods", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--download-workers", type=int, default=16)
    ap.add_argument("--pod-prefix", default=PREFIX)
    ap.add_argument("--clap-model", default=CLAP_MODEL)
    ap.add_argument("--poll-sec", type=int, default=45)
    args = ap.parse_args()

    try:
        return run_ramp(args)
    except Exception as exc:  # noqa: BLE001
        print(f"RAMP_FATAL {type(exc).__name__}: {exc}", flush=True)
        raise


def run_ramp(args: argparse.Namespace) -> int:
    key = os.environ["RUNPOD_API_KEY"].strip()
    token = os.environ["HF_TOKEN"].strip()
    root = Path(__file__).resolve().parent
    data = root.parent / "text2asmr" / "data"
    script_b64 = base64.b64encode((root / "clap_label_audios2.py").read_bytes()).decode()
    ontology_b64 = base64.b64encode((data / "ontology.py").read_bytes()).decode()
    segment_b64 = base64.b64encode((data / "segment.py").read_bytes()).decode()
    pub = (Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()

    started = time.time()
    launched: dict[int, dict] = {}  # shard -> pod
    ready_streak: dict[str, int] = {}
    idle: dict[str, int] = {}
    next_shard = 0
    bad_machines: set[str] = set()
    cuda_fails = 0

    # Prefer Secure after community CUDA passthrough failures; Secure is ~2x $/h
    # but actually usable. Start Secure immediately if COMMUNITY_FORCE unset.
    if os.environ.get("CLAP_CLOUD", "SECURE").upper() == "SECURE":
        launch.CLOUD_TYPE = "SECURE"
        print("CLOUD=SECURE (reliable GPU mapping)", flush=True)
    else:
        launch.CLOUD_TYPE = "COMMUNITY"
        print("CLOUD=COMMUNITY", flush=True)

    balance, existing = list_clap_pods(key)
    print(
        f"START balance=${balance:.2f} existing={len(existing)} "
        f"model={args.clap_model} batch={args.batch} workers={args.download_workers} "
        f"max_pods={args.max_pods} shards={args.shards}",
        flush=True,
    )
    print(
        "ETA: best ~6–12h wall with up to 8×4090; worst ~24h. "
        f"Spend cap ${MAX_SPEND_USD:.0f}; stop if balance < ${BALANCE_FLOOR:.0f}. "
        f"cloud={launch.CLOUD_TYPE}.",
        flush=True,
    )

    # Resume indexing from existing names
    for p in existing:
        m = re.fullmatch(rf"{re.escape(args.pod_prefix)}(\d+)", p.get("name") or "")
        if m:
            idx = int(m.group(1))
            launched[idx] = p
            next_shard = max(next_shard, idx + 1)

    while True:
        balance, pods = list_clap_pods(key)
        by_id = {p["id"]: p for p in pods}
        # drop terminated from launched
        launched = {i: p for i, p in launched.items() if p.get("id") in by_id}
        live = [by_id[p["id"]] for p in launched.values() if p.get("id") in by_id]
        burn = sum(float(p.get("costPerHr") or 0) for p in live)
        elapsed_h = max((time.time() - started) / 3600.0, 1e-6)
        spent = burn * elapsed_h
        print(
            f"WATCH live={len(live)} next_shard={next_shard}/{args.shards} "
            f"burn=${burn:.2f}/h spent_est=${spent:.2f} balance=${balance:.2f}",
            flush=True,
        )

        if spent >= MAX_SPEND_USD or balance <= BALANCE_FLOOR:
            for p in live:
                terminate(key, p["id"])
                print(f"TERMINATED_BUDGET {p['name']} {p['id']}", flush=True)
            print("FLEET_STOPPED_BUDGET", flush=True)
            return 2

        # Terminate completed / idle; ensure bootstrap on live pods
        for p in list(live):
            host = (p.get("machine") or {}).get("podHostId")
            runtime = p.get("runtime") or {}
            up = float(runtime.get("uptimeInSeconds") or 0)
            g = (runtime.get("gpus") or [{}])[0]
            util = float(g.get("gpuUtilPercent") or 0)
            mem = float(g.get("memoryUtilPercent") or 0)
            try:
                boot = ensure_bootstrap(host)
            except Exception as exc:  # noqa: BLE001
                boot = f"ERR:{type(exc).__name__}"
            print(
                f"  {p['name']} util={util:.0f}% mem={mem:.0f}% up={up/60:.0f}m boot={boot}",
                flush=True,
            )
            try:
                done = pod_done(host)
            except Exception as exc:  # noqa: BLE001
                print(f"  pod_done_err {p['name']}: {exc}", flush=True)
                done = False
            if done:
                terminate(key, p["id"])
                print(f"TERMINATED_DONE {p['name']}", flush=True)
                continue
            # Broken GPU passthrough (common on community): recycle same shard on a new host.
            if up > 150 and util < 2:
                try:
                    bad = pod_cuda_bad(host)
                except Exception as exc:  # noqa: BLE001
                    print(f"  cuda_check_err {p['name']}: {exc}", flush=True)
                    bad = False
                if bad:
                    m = re.fullmatch(
                        rf"{re.escape(args.pod_prefix)}(\d+)", p.get("name") or ""
                    )
                    shard_idx = int(m.group(1)) if m else None
                    mid = p.get("machineId") or ""
                    if mid:
                        bad_machines.add(mid)
                    cuda_fails += 1
                    if cuda_fails >= 2 and launch.CLOUD_TYPE != "SECURE":
                        launch.CLOUD_TYPE = "SECURE"
                        print("SWITCH_CLOUD=SECURE after CUDA failures", flush=True)
                    terminate(key, p["id"])
                    if shard_idx is not None:
                        launched.pop(shard_idx, None)
                        next_shard = min(next_shard, shard_idx)
                    print(
                        f"TERMINATED_CUDA_RECYCLE {p['name']} retry_shard={shard_idx} "
                        f"bad_machine={mid}",
                        flush=True,
                    )
                    continue
            # Don't idle-kill a live labeler: Hub download/upload can hold GPU at 0%
            # for many minutes between tagging bursts.
            if up > 900 and util < 2 and boot not in {"RUNNING", "STARTING"}:
                idle[p["id"]] = idle.get(p["id"], 0) + 1
            else:
                idle[p["id"]] = 0
            if idle.get(p["id"], 0) >= 4:
                terminate(key, p["id"])
                print(f"TERMINATED_IDLE {p['name']}", flush=True)

        balance, pods = list_clap_pods(key)
        by_id = {p["id"]: p for p in pods}
        live = [by_id[p["id"]] for p in launched.values() if p.get("id") in by_id]

        # Saturation gate for newest pod
        can_launch = False
        if not live and next_shard < args.shards:
            can_launch = True
        elif live and next_shard < args.shards and len(live) < args.max_pods:
            def shard_of(p):
                m = re.fullmatch(rf"{re.escape(args.pod_prefix)}(\d+)", p.get("name") or "")
                return int(m.group(1)) if m else -1
            newest = max(live, key=shard_of)
            if pod_saturated(newest):
                ready_streak[newest["id"]] = ready_streak.get(newest["id"], 0) + 1
            else:
                ready_streak[newest["id"]] = 0
            if ready_streak.get(newest["id"], 0) >= 2:
                can_launch = True
                print(
                    f"READY {newest['name']} saturated x2 — launching next",
                    flush=True,
                )

        if can_launch and next_shard < args.shards and len(live) < args.max_pods:
            # Prefer filling holes (e.g. shard 0 killed) before advancing.
            live_idxs = set()
            for p in live:
                m = re.fullmatch(rf"{re.escape(args.pod_prefix)}(\d+)", p.get("name") or "")
                if m:
                    live_idxs.add(int(m.group(1)))
            holes = [i for i in range(max(next_shard, 1)) if i not in live_idxs]
            idx = holes[0] if holes else next_shard
            try:
                pod = create_shard(
                    key, token, pub, script_b64, ontology_b64, segment_b64,
                    args.shards, idx, args.pod_prefix, args.batch,
                    args.download_workers, args.clap_model,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL shard={idx}: {exc}", flush=True)
                time.sleep(args.poll_sec)
                continue
            if not pod:
                print(f"FAIL shard={idx}: null pod", flush=True)
                time.sleep(args.poll_sec)
                continue
            mid = pod.get("machineId") or ""
            if mid and mid in bad_machines:
                terminate(key, pod["id"])
                print(f"TERMINATED_BAD_MACHINE shard={idx} machine={mid}", flush=True)
                time.sleep(5)
                continue
            launched[idx] = pod
            next_shard = max(next_shard, idx + 1)
            print(
                f"CREATED shard={idx} id={pod['id']} ${pod.get('costPerHr')}/h "
                f"cloud={launch.CLOUD_TYPE} machine={mid}",
                flush=True,
            )
            save_state(list(launched.values()), args.shards, args.pod_prefix, started)
            # Non-blocking: bootstrap is kicked each WATCH poll once host exists.
            print(f"  queued_bootstrap shard={idx} (will kick on host)", flush=True)

        # All shards launched and all finished?
        balance, pods = list_clap_pods(key)
        live = [p for p in pods if (p.get("name") or "").startswith(args.pod_prefix)]
        if next_shard >= args.shards and not live:
            print("FLEET_DONE all shards finished", flush=True)
            return 0

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
