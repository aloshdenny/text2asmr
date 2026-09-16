#!/usr/bin/env python3
"""Watch t2a-clap-ft: on CLAP_FT_DONE, confirm Hub push (or scp), terminate pod."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

GQL = "https://api.runpod.io/graphql"


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}", flush=True)


def gql(key: str, query: str, variables: dict | None = None) -> dict:
    body: dict = {"query": query}
    if variables is not None:
        body["variables"] = variables
    req = urllib.request.Request(
        f"{GQL}?api_key={key}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        result = json.loads(resp.read())
    if result.get("errors"):
        raise RuntimeError(result["errors"])
    return result["data"]


def terminate(key: str, pod_id: str) -> None:
    log(f"terminating {pod_id}")
    data = gql(
        key,
        "mutation($id:String!){ podTerminate(input:{podId:$id}) }",
        {"id": pod_id},
    )
    log(f"resp {json.dumps(data)[:300]}")


def pod_runtime(key: str, pod_id: str) -> dict | None:
    data = gql(
        key,
        "query { myself { currentSpendPerHr pods { id name desiredStatus costPerHr "
        "runtime { ports { ip isIpPublic privatePort publicPort type } uptimeInSeconds } "
        "machine { podHostId } } } }",
    )
    for p in data["myself"]["pods"]:
        if p["id"] == pod_id:
            return p
    return None


def ssh_ports(pod: dict) -> tuple[str, int] | None:
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port.get("type") == "tcp" and int(port.get("privatePort") or 0) == 22 and port.get("isIpPublic"):
            return port["ip"], int(port["publicPort"])
    return None


def ssh_cmd(host: str, port: int, key_path: str, remote: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh", "-p", str(port),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=25", "-i", key_path, f"root@{host}", remote,
        ],
        capture_output=True, text=True, timeout=90,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-id", required=True)
    ap.add_argument("--ssh-key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    ap.add_argument("--poll-sec", type=int, default=120)
    ap.add_argument("--eta-sec", type=int, default=3600)
    ap.add_argument("--local-out", type=Path, default=Path("ckpt/clap-ft"))
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY") or ""
    if not key:
        log("RUNPOD_API_KEY missing")
        return 2

    idle = 0
    next_eta = time.time() + args.eta_sec
    log(f"watching clap-ft pod={args.pod_id}")
    log("ETA: best 45–90m, worst ~3h")

    while True:
        pod = pod_runtime(key, args.pod_id)
        if pod is None:
            log("pod gone — exit")
            return 0
        ports = ssh_ports(pod)
        if not ports:
            log("waiting for SSH port...")
            time.sleep(args.poll_sec)
            continue
        host, port = ports
        r = ssh_cmd(
            host, port, args.ssh_key,
            "test -f /workspace/clap-ft/CHECKPOINT_DONE && echo DONE=1 || echo DONE=0; "
            "pgrep -fc finetune_clap.py || true; "
            "tail -n 3 /workspace/bootstrap.log 2>/dev/null || true",
        )
        out = (r.stdout or "") + (r.stderr or "")
        log(out.replace("\n", " | ")[:400])
        done = "DONE=1" in out
        running = "finetune_clap" in out and "DONE=0" in out
        if done:
            # pull checkpoint locally as backup
            args.local_out.mkdir(parents=True, exist_ok=True)
            log(f"pulling checkpoint -> {args.local_out}")
            subprocess.run(
                [
                    "scp", "-P", str(port),
                    "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                    "-i", args.ssh_key, "-r",
                    f"root@{host}:/workspace/clap-ft/",
                    str(args.local_out),
                ],
                check=False,
            )
            terminate(key, args.pod_id)
            log("watcher exit")
            return 0
        if not running and "BOOT" in out:
            idle += 1
        else:
            idle = 0
        if idle >= 3:
            log("idle with no finetune — terminate")
            terminate(key, args.pod_id)
            return 1
        if time.time() >= next_eta:
            log("ETA re-check: still training")
            next_eta = time.time() + args.eta_sec
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
