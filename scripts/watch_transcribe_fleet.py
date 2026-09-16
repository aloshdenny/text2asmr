#!/usr/bin/env python3
"""Terminate completed/idle transcription pods and protect the spend cap."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

GQL = "https://api.runpod.io/graphql"
STATE = Path("/tmp/t2a_fleet_watch_state.json")
def gql(query: str, key: str) -> dict:
    request = urllib.request.Request(
        f"{GQL}?api_key={key}",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 t2a-watch"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(result["errors"])
    return result["data"]


def terminate(key: str, pod_id: str) -> None:
    gql(f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}', key)


def list_hub_files() -> list[str]:
    code = """
import json
import os
from huggingface_hub import HfApi
print(json.dumps(HfApi(token=os.environ["HF_TOKEN"]).list_repo_files(
    "aoxo/audios2", repo_type="dataset"
)))
"""
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        text=True,
        timeout=180,
    )
    return json.loads(output)


def transcription_running(host_id: str | None) -> bool:
    if not host_id:
        return False
    try:
        result = subprocess.run(
            [
                "ssh", "-tt", "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10",
                f"{host_id}@ssh.runpod.io",
            ],
            input=(
                "pgrep -f '[t]ranscribe_audios2.py' >/dev/null; "
                "echo TRANSCRIBE_STATUS_$?; exit\n"
            ),
            text=True,
            capture_output=True,
            timeout=20,
        )
        # RunPod's forced PTY echoes stdin. The echoed command contains literal
        # "$?", while only the executed output contains the numeric result.
        return "TRANSCRIBE_STATUS_0" in result.stdout
    except (OSError, subprocess.SubprocessError):
        return True  # Never terminate a pod merely because the safety probe failed.


def start_shard(
    pod: dict,
    shard: int,
    num_shards: int,
    creator_flag: str,
) -> bool:
    machine = pod.get("machine") or {}
    host_id = machine.get("podHostId")
    if not host_id:
        return False
    workers = 8 if "A100" in (machine.get("gpuDisplayName") or "") else 3
    command = f"""
export TRANSCRIBE_BASE=/workspace/t2a
nohup python /workspace/transcribe_audios2.py \
  --model large-v3 --compute-type float16 \
  --transcribe-workers {workers} --producer-workers {workers} --uploader-workers 1 \
  --upload-batch-size 128 \
  {creator_flag} /workspace/tinkerspace_creators.txt \
  --num-shards {num_shards} --shard-index {shard} --shard-key file \
  > /workspace/shard-{shard:02d}.log 2>&1 &
echo SHARD_STARTED_{shard}
exit
"""
    try:
        result = subprocess.run(
            [
                "ssh", "-tt", "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10",
                f"{host_id}@ssh.runpod.io",
            ],
            input=command,
            text=True,
            capture_output=True,
            timeout=25,
        )
        return bool(
            re.search(
                rf"(?m)^SHARD_STARTED_{shard}\r?$",
                result.stdout,
            )
        )
    except (OSError, subprocess.SubprocessError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-prefix", default="t2a-fast-")
    parser.add_argument("--num-shards", type=int, default=60)
    parser.add_argument(
        "--include-creators",
        action="store_true",
        help="watch only creators in transcribe_shards/tinkerspace.txt",
    )
    args = parser.parse_args()
    creators = {
        line.strip()
        for line in (
            Path(__file__).parent / "transcribe_shards" / "tinkerspace.txt"
        ).read_text().splitlines()
        if line.strip()
    }
    creator_flag = "--creators-file" if args.include_creators else "--exclude-creators-file"
    key = os.environ["RUNPOD_API_KEY"]
    os.environ["HF_TOKEN"]  # Fail immediately if the required credential is absent.
    state = json.loads(STATE.read_text()) if STATE.exists() else {"idle": {}, "assigned": {}}
    idle = state.get("idle", {})
    assigned = state.get("assigned", {})
    last_hub_check = 0.0

    while True:
        now = time.time()
        data = gql(
            """query { myself { clientBalance pods {
                id name desiredStatus costPerHr
                machine { podHostId gpuDisplayName }
                runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } }
            } } }""",
            key,
        )["myself"]
        pods = [
            p for p in data.get("pods") or []
            if p["name"].startswith(args.pod_prefix)
        ]
        live_ids = {p["id"] for p in pods}
        assigned = {pod_id: shard for pod_id, shard in assigned.items() if pod_id in live_ids}
        for pod in pods:
            if pod["id"] not in assigned:
                match = re.fullmatch(re.escape(args.pod_prefix) + r"(\d+)", pod["name"])
                if match:
                    assigned[pod["id"]] = int(match.group(1))
        burn = sum(float(p.get("costPerHr") or 0) for p in pods)
        balance = float(data.get("clientBalance") or 0)
        print(
            f"WATCH pods={len(pods)} burn=${burn:.2f}/h balance=${balance:.2f}",
            flush=True,
        )

        if balance <= 3:
            for pod in pods:
                terminate(key, pod["id"])
                print(f"TERMINATED_BUDGET {pod['name']} {pod['id']}", flush=True)
            print("FLEET_STOPPED_BUDGET", flush=True)
            return 2

        force_hub_check = False
        for pod in pods:
            runtime = pod.get("runtime") or {}
            uptime = float(runtime.get("uptimeInSeconds") or 0)
            gpu = (runtime.get("gpus") or [{}])[0]
            util = float(gpu.get("gpuUtilPercent") or 0)
            vram = float(gpu.get("memoryUtilPercent") or 0)
            pod_id = pod["id"]
            idle[pod_id] = idle.get(pod_id, 0) + 1 if uptime > 900 and util < 2 else 0
            print(
                f"GPU {pod['name']} util={util:.0f}% vram={vram:.0f}% "
                f"up={uptime / 60:.0f}m idle={idle[pod_id]}",
                flush=True,
            )
            if idle[pod_id] >= 3:
                host_id = (pod.get("machine") or {}).get("podHostId")
                if transcription_running(host_id):
                    idle[pod_id] = 0
                    print(f"KEPT_ACTIVE {pod['name']} {pod_id}", flush=True)
                    continue
                # Check whether this pod finished its shard before terminating
                # it as failed. A completed pod should be rotated to an
                # uncovered shard, not discarded by the faster idle timer.
                force_hub_check = True

        if force_hub_check or now - last_hub_check >= 600:
            try:
                files = list_hub_files()
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                print(f"HUB_CHECK_FAILED {type(exc).__name__}: {exc}", flush=True)
                last_hub_check = now
                STATE.write_text(
                    json.dumps({"idle": idle, "assigned": assigned, "checked": now})
                )
                time.sleep(120)
                continue
            all_files = set(files)
            pending = [
                path
                for path in files
                if path.lower().endswith(".m4a")
                and path + ".json" not in all_files
                and (
                    path.split("/")[0] in creators
                    if args.include_creators
                    else path.split("/")[0] not in creators
                )
            ]
            per_shard = [0] * args.num_shards
            for path in pending:
                per_shard[zlib.crc32(path.encode()) % args.num_shards] += 1
            active_shards = {int(shard) for shard in assigned.values()}
            missing = [
                index
                for index, count in enumerate(per_shard)
                if count and index not in active_shards
            ]
            print(
                f"HUB runpod_pending={len(pending)} missing_shards={missing}",
                flush=True,
            )
            if not pending:
                for pod in pods:
                    terminate(key, pod["id"])
                    print(f"TERMINATED_DONE {pod['name']} {pod['id']}", flush=True)
                print("FLEET_DONE", flush=True)
                return 0

            for pod in pods:
                current_shard = int(assigned.get(pod["id"], -1))
                if current_shard < 0:
                    continue
                if per_shard[current_shard] != 0:
                    if idle.get(pod["id"], 0) >= 3:
                        host_id = (pod.get("machine") or {}).get("podHostId")
                        if not transcription_running(host_id):
                            terminate(key, pod["id"])
                            print(
                                f"TERMINATED_IDLE {pod['name']} {pod['id']}",
                                flush=True,
                            )
                    continue
                host_id = (pod.get("machine") or {}).get("podHostId")
                if transcription_running(host_id):
                    continue
                if missing:
                    next_shard = missing.pop(0)
                    if start_shard(
                        pod,
                        next_shard,
                        args.num_shards,
                        creator_flag,
                    ):
                        assigned[pod["id"]] = next_shard
                        print(
                            f"REASSIGNED {pod['name']} shard={current_shard}->{next_shard}",
                            flush=True,
                        )
                        continue
                else:
                    terminate(key, pod["id"])
                    print(f"TERMINATED_DONE {pod['name']} {pod['id']}", flush=True)
            last_hub_check = now

        STATE.write_text(json.dumps({"idle": idle, "assigned": assigned, "checked": now}))
        time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())
