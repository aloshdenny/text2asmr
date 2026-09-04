#!/usr/bin/env python3
"""Report RunPod pod state and spend, and flag pods that are burning money.

Designed to be run on a timer. GPU pods bill continuously whether or not they
are doing anything, and the expensive failure is not a crash -- it is a pod
that looks alive while its job has finished, died, or silently no-opped.

    source ~/.t2a_env && python scripts/runpod_watch.py
    source ~/.t2a_env && python scripts/runpod_watch.py --terminate-idle

Exit codes: 0 healthy or nothing running, 1 attention needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GQL = "https://api.runpod.io/graphql"
STATE = Path.home() / ".cache" / "text2asmr-runpod-watch.json"


def gql(query: str, api_key: str) -> dict:
    req = urllib.request.Request(
        f"{GQL}?api_key={api_key}",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"RunPod API error {e.code}: {e.read()[:300].decode()}")
    if "errors" in out:
        raise SystemExit(f"RunPod GraphQL error: {out['errors']}")
    return out["data"]


def fetch_pods(api_key: str) -> list[dict]:
    q = """
    query { myself { pods {
        id name desiredStatus costPerHr lastStatusChange
        runtime { uptimeInSeconds
                  gpus { gpuUtilPercent memoryUtilPercent }
                  container { cpuPercent memoryPercent } }
    } } }
    """
    return gql(q, api_key)["myself"]["pods"] or []


def terminate(api_key: str, pod_id: str) -> None:
    gql(f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}', api_key)


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle-threshold", type=float, default=5.0,
                    help="GPU utilisation %% below which a pod counts as idle")
    ap.add_argument("--idle-checks", type=int, default=3,
                    help="consecutive idle observations before flagging")
    ap.add_argument("--terminate-idle", action="store_true",
                    help="actually terminate pods flagged idle (destructive)")
    ap.add_argument("--budget", type=float, default=50.0,
                    help="warn once cumulative spend passes this")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print("RUNPOD_API_KEY not set (source ~/.t2a_env first)", file=sys.stderr)
        return 1

    pods = fetch_pods(api_key)
    state = load_state()
    now = time.time()
    attention = False

    if not pods:
        print("no pods running - nothing billing")
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"checked": now, "pods": {}}))
        return 0

    tracked: dict = {}
    total_rate = 0.0
    for pod in pods:
        pid = pod["id"]
        rate = float(pod.get("costPerHr") or 0)
        rt = pod.get("runtime") or {}
        up_s = float(rt.get("uptimeInSeconds") or 0)
        gpus = rt.get("gpus") or []
        util = max((float(g.get("gpuUtilPercent") or 0) for g in gpus),
                   default=0.0)
        spend = rate * up_s / 3600
        total_rate += rate if pod.get("desiredStatus") == "RUNNING" else 0

        prev = state.get("pods", {}).get(pid, {})
        idle_streak = prev.get("idle_streak", 0)
        # Only count idleness while the pod is actually up; a pod still
        # booting reports 0% and is not wasting anything yet.
        if pod.get("desiredStatus") == "RUNNING" and up_s > 300:
            idle_streak = idle_streak + 1 if util < args.idle_threshold else 0
        tracked[pid] = {"idle_streak": idle_streak, "spend": spend}

        flag = ""
        if idle_streak >= args.idle_checks:
            flag = f"  <-- IDLE for {idle_streak} checks, still billing"
            attention = True
        elif spend > args.budget:
            flag = "  <-- over budget"
            attention = True

        print(f"{pod['name']} [{pid}] {pod.get('desiredStatus')}  "
              f"up {up_s/3600:.1f}h  gpu {util:.0f}%  "
              f"${rate:.2f}/hr  spent ~${spend:.2f}{flag}")

        if flag and args.terminate_idle and idle_streak >= args.idle_checks:
            terminate(api_key, pid)
            print(f"  terminated {pid}")

    print(f"total burn rate: ${total_rate:.2f}/hr "
          f"(${total_rate*24:.2f}/day if left running)")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"checked": now, "pods": tracked}))
    return 1 if attention else 0


if __name__ == "__main__":
    raise SystemExit(main())
