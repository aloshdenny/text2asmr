#!/usr/bin/env python3
"""Terminate completed/idle CLAP-label pods. Hard spend guard for ~$5 budget."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.request

GQL = "https://api.runpod.io/graphql"
PREFIX = "t2a-clap-"
# Soft budget: terminate whole fleet if burn*elapsed would exceed this.
MAX_SPEND_USD = 5.0
STATE = "/tmp/t2a_clap_fleet.json"


def gql(query: str, key: str) -> dict:
    request = urllib.request.Request(
        f"{GQL}?api_key={key}",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 t2a-clap-watch"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(result["errors"])
    return result["data"]


def terminate(key: str, pod_id: str) -> None:
    gql(f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}', key)


def ssh_done(host_id: str | None) -> bool:
    if not host_id:
        return False
    try:
        result = subprocess.run(
            [
                "ssh", "-tt", "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10",
                "-i", str(Path.home() / ".ssh" / "id_ed25519"),
                f"{host_id}@ssh.runpod.io",
            ],
            input="grep -E 'CLAP_DONE|CLAP_DONE shard=' /workspace/bootstrap.log 2>/dev/null | tail -n 1; "
                  "pgrep -af '[c]lap_label_audios2' || true; echo STATUS_DONE; exit\n",
            text=True, capture_output=True, timeout=25,
        )
        out = result.stdout
        if re.search(r"(?m)^CLAP_DONE", out) or "CLAP_DONE shard=" in out:
            return True
        if "clap_label_audios2" not in out and "STATUS_DONE" in out:
            # process gone; treat as done if log says so, else idle fail
            return "CLAP_DONE" in out
    except (OSError, subprocess.SubprocessError):
        return False
    return False


from pathlib import Path  # noqa: E402


def main() -> int:
    key = os.environ["RUNPOD_API_KEY"]
    started = time.time()
    if Path(STATE).exists():
        started = json.loads(Path(STATE).read_text()).get("created_at", started)
    idle: dict[str, int] = {}

    while True:
        data = gql(
            """query { myself { clientBalance pods {
                id name costPerHr
                machine { podHostId }
                runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } }
            } } }""",
            key,
        )["myself"]
        pods = [p for p in data.get("pods") or [] if (p.get("name") or "").startswith(PREFIX)]
        burn = sum(float(p.get("costPerHr") or 0) for p in pods)
        elapsed_h = max((time.time() - started) / 3600.0, 1e-6)
        spent_est = burn * elapsed_h
        balance = float(data.get("clientBalance") or 0)
        print(
            f"WATCH pods={len(pods)} burn=${burn:.2f}/h spent_est=${spent_est:.2f} "
            f"balance=${balance:.2f}",
            flush=True,
        )

        if not pods:
            print("FLEET_DONE", flush=True)
            return 0

        if spent_est >= MAX_SPEND_USD or balance <= 3:
            for pod in pods:
                terminate(key, pod["id"])
                print(f"TERMINATED_BUDGET {pod['name']} {pod['id']}", flush=True)
            print("FLEET_STOPPED_BUDGET", flush=True)
            return 2

        for pod in pods:
            runtime = pod.get("runtime") or {}
            uptime = float(runtime.get("uptimeInSeconds") or 0)
            gpu = (runtime.get("gpus") or [{}])[0]
            util = float(gpu.get("gpuUtilPercent") or 0)
            host = (pod.get("machine") or {}).get("podHostId")
            print(
                f"GPU {pod['name']} util={util:.0f}% up={uptime/60:.0f}m idle={idle.get(pod['id'],0)}",
                flush=True,
            )
            if ssh_done(host):
                terminate(key, pod["id"])
                print(f"TERMINATED_DONE {pod['name']} {pod['id']}", flush=True)
                continue
            if uptime > 900 and util < 2:
                idle[pod["id"]] = idle.get(pod["id"], 0) + 1
            else:
                idle[pod["id"]] = 0
            if idle.get(pod["id"], 0) >= 4:
                terminate(key, pod["id"])
                print(f"TERMINATED_IDLE {pod['name']} {pod['id']}", flush=True)

        time.sleep(90)


if __name__ == "__main__":
    raise SystemExit(main())
