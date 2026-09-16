#!/usr/bin/env python3
"""Watch t2a-gemini-cpu: pull label_tool artifacts, then terminate the pod.

Usage:
  source ~/.t2a_env
  nohup python3 -u scripts/watch_gemini_cpu_pod.py --pod-id j6nov8ibrn0nsa \\
    --ssh-host 38.80.152.147 --ssh-port 39130 \\
    > label_tool/watch_gemini_cpu.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LOCAL_OUT = HERE / "label_tool" / "from_pod"
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
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GraphQL HTTP {e.code}: {e.read()[:300]!r}") from e


def terminate(key: str, pod_id: str) -> None:
    log(f"terminating pod {pod_id}")
    data = gql(
        key,
        "mutation($id:String!){ podTerminate(input:{podId:$id}) }",
        {"id": pod_id},
    )
    log(f"terminate response: {json.dumps(data)[:400]}")


def ssh_cmd(host: str, port: int, key_path: str, remote: str) -> subprocess.CompletedProcess:
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
            "-i",
            key_path,
            f"root@{host}",
            remote,
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )


def pull(host: str, port: int, key_path: str) -> None:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    dest = LOCAL_OUT / time.strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    log(f"pulling label_tool -> {dest}")
    r = subprocess.run(
        [
            "scp",
            "-P",
            str(port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-i",
            key_path,
            "-r",
            f"root@{host}:/workspace/text2asmr/label_tool/",
            str(dest),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode != 0:
        log(f"scp stderr: {(r.stderr or '')[-500]}")
        raise RuntimeError(f"scp failed rc={r.returncode}")
    # also merge into main label_tool for convenience
    merge = HERE / "label_tool"
    for pat in ("gemini_audios2.shard*.jsonl", "clap_train_audios2.shard*.jsonl"):
        for p in dest.glob(f"**/{pat}"):
            target = merge / p.name
            target.write_bytes(p.read_bytes())
            log(f"copied {p.name} ({p.stat().st_size} bytes)")
    log("pull ok")


def restart_label(host: str, port: int, key_path: str) -> None:
    remote = (
        "if pgrep -f supervise_gemini_label.sh >/dev/null; then echo SUP_ALIVE; exit 0; fi; "
        "nohup bash /workspace/text2asmr/scripts/supervise_gemini_label.sh "
        ">> /workspace/supervise.log 2>&1 & echo RESTARTED $!"
    )
    r = ssh_cmd(host, port, key_path, remote)
    log(f"restart {(r.stdout or '')[-200:].strip()} rc={r.returncode}")


def remote_status(host: str, port: int, key_path: str) -> dict:
    remote = (
        "cd /workspace/text2asmr; "
        "echo DONE=$(test -f label_tool/GEMINI_ALL_DONE && echo 1 || echo 0); "
        "echo PIDS=$(pgrep -fc annotate_audios2_gemini_batch.py || true); "
        "echo SUP=$(pgrep -fc supervise_gemini_label.sh || true); "
        "echo LINES=$(wc -l < label_tool/gemini_audios2.batch.vertex.jsonl 2>/dev/null || echo 0); "
        "echo LAST=$(tail -n 1 /workspace/gemini_vertex_batch.log 2>/dev/null | tr '\\n' ' ')"
    )
    r = ssh_cmd(host, port, key_path, remote)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    if r.returncode != 0 and "DONE=" not in out:
        return {"ok": False, "raw": out[-800:], "rc": r.returncode}
    info: dict = {"ok": True, "raw": out}
    for line in out.splitlines():
        if "=" in line and line.split("=", 1)[0] in {"DONE", "PIDS", "SUP", "LINES", "LAST"}:
            k, v = line.split("=", 1)
            info[k] = v.strip()
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-id", required=True)
    ap.add_argument("--ssh-host", required=True)
    ap.add_argument("--ssh-port", type=int, required=True)
    ap.add_argument("--ssh-key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    ap.add_argument("--poll-sec", type=int, default=300)
    ap.add_argument("--eta-sec", type=int, default=7200, help="re-check ETA / progress interval")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY") or ""
    if not key:
        log("RUNPOD_API_KEY missing")
        return 2

    next_eta = time.time() + args.eta_sec
    log(
        f"watching pod={args.pod_id} host={args.ssh_host}:{args.ssh_port} "
        f"poll={args.poll_sec}s eta_every={args.eta_sec}s"
    )
    log("ETA: best ~2–4d, worst ~8–12d; restart annotator on death; terminate only on GEMINI_ALL_DONE")

    while True:
        try:
            st = remote_status(args.ssh_host, args.ssh_port, args.ssh_key)
        except Exception as exc:
            log(f"status error: {type(exc).__name__}: {exc}")
            time.sleep(args.poll_sec)
            continue

        if not st.get("ok"):
            log(f"ssh bad rc={st.get('rc')} {(st.get('raw') or '')[-300]}")
            # SSH flaps are not idle and are not a terminate signal.
        else:
            done = st.get("DONE") == "1"
            try:
                pids = int(st.get("PIDS") or "0")
            except ValueError:
                pids = 0
            try:
                sup = int(st.get("SUP") or "0")
            except ValueError:
                sup = 0
            lines = st.get("LINES", "?")
            log(
                f"status done={done} annotator={pids} supervise={sup} "
                f"vertex_lines={lines} last={st.get('LAST','')[:120]}"
            )

            if done:
                log("GEMINI_ALL_DONE — pulling then terminating")
                try:
                    pull(args.ssh_host, args.ssh_port, args.ssh_key)
                except Exception as exc:
                    log(f"pull failed ({exc}); terminating anyway to stop burn")
                terminate(key, args.pod_id)
                log("watcher exit")
                return 0

            if pids == 0 or sup == 0:
                log("labelling/supervise missing — restarting on pod")
                try:
                    restart_label(args.ssh_host, args.ssh_port, args.ssh_key)
                except Exception as exc:
                    log(f"restart failed: {type(exc).__name__}: {exc}")

        if time.time() >= next_eta:
            log("ETA re-check: still running; remaining ~same order as start unless rate improves")
            try:
                pull(args.ssh_host, args.ssh_port, args.ssh_key)
            except Exception as exc:
                log(f"interim pull failed: {exc}")
            next_eta = time.time() + args.eta_sec

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
