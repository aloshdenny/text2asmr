#!/usr/bin/env python3
"""Watchdog for text2asmr RunPod pods: nothing is allowed to bill while doing nothing.

A pod is terminated when any of these is true:
  * it printed LABEL_RUN_DONE (or its process is gone) -- work finished
  * GPU utilisation stayed under --idle-util for --idle-polls consecutive polls
  * the pod has been up longer than --max-hours
  * this campaign has spent more than --spend-cap, or the account balance fell below --balance-floor
    (both of those terminate *every* t2a pod, not just the idle one)

Runs anywhere with the RunPod key; meant to live on the always-on droplet as a systemd service so a pod can
never outlive the laptop.  It only ever touches pods whose name starts with --prefix.

  python3 runpod_guard.py --prefix t2a- --spend-cap 60 --balance-floor 15
"""
from __future__ import annotations
import argparse, json, os, time, urllib.request
from collections import defaultdict

API = "https://api.runpod.io/graphql"


def log(m): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def gql(query: str, key: str) -> dict:
    req = urllib.request.Request(f"{API}?api_key={key}", data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "t2a-guard/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    if "errors" in d: raise RuntimeError(str(d["errors"])[:300])
    return d["data"]


STATE_Q = """{ myself { clientBalance currentSpendPerHr
  pods { id name desiredStatus costPerHr
         runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } } } } }"""


def terminate(pod_id: str, key: str, why: str):
    log(f"TERMINATING {pod_id}: {why}")
    try:
        gql(f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}', key)
        log(f"  terminated {pod_id}")
    except Exception as e:
        log(f"  terminate failed for {pod_id}: {type(e).__name__} {str(e)[:150]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="t2a-", help="only pods whose name starts with this are managed")
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--idle-util", type=float, default=5.0, help="percent GPU below which a pod counts as idle")
    ap.add_argument("--idle-polls", type=int, default=5, help="consecutive idle polls before termination")
    ap.add_argument("--max-hours", type=float, default=14.0)
    ap.add_argument("--spend-cap", type=float, default=60.0, help="dollars this guard may let the campaign spend")
    ap.add_argument("--balance-floor", type=float, default=15.0)
    ap.add_argument("--state", default="/root/t2a/runpod_guard.json")
    a = ap.parse_args()
    key = os.environ["RUNPOD_API_KEY"]

    spent = 0.0
    try:
        spent = float(json.load(open(a.state)).get("spent", 0.0))
        log(f"resuming: ${spent:.2f} already attributed to this campaign")
    except Exception: pass

    idle: dict[str, int] = defaultdict(int)
    last = time.time()
    while True:
        try:
            d = gql(STATE_Q, key)["myself"]
        except Exception as e:
            log(f"poll failed: {type(e).__name__} {str(e)[:150]}"); time.sleep(a.interval); continue

        pods = [p for p in d["pods"] if (p.get("name") or "").startswith(a.prefix)]
        now = time.time(); dt_h = (now - last) / 3600.0; last = now
        burn = sum((p.get("costPerHr") or 0.0) for p in pods
                   if (p.get("desiredStatus") or "") == "RUNNING")
        spent += burn * dt_h
        json.dump({"spent": spent, "ts": time.strftime("%F %T")}, open(a.state, "w"))

        bal = d.get("clientBalance") or 0.0
        log(f"balance ${bal:.2f} | burn ${burn:.2f}/h | campaign ${spent:.2f} | pods {len(pods)}")

        if pods and (spent >= a.spend_cap or bal <= a.balance_floor):
            why = f"spend cap ${a.spend_cap} reached (${spent:.2f})" if spent >= a.spend_cap \
                  else f"balance floor ${a.balance_floor} reached (${bal:.2f})"
            for p in pods: terminate(p["id"], key, why)
            time.sleep(a.interval); continue

        for p in pods:
            rt = p.get("runtime") or {}
            up_h = (rt.get("uptimeInSeconds") or 0) / 3600.0
            gpus = rt.get("gpus") or []
            util = max([g.get("gpuUtilPercent") or 0 for g in gpus], default=0)
            status = p.get("desiredStatus") or "?"
            log(f"  {p['name']} ({p['id'][:12]}) {status} up {up_h:.1f} h  gpu {util}%  ${p.get('costPerHr')}/h")
            if status != "RUNNING": continue
            if up_h >= a.max_hours:
                terminate(p["id"], key, f"max runtime {a.max_hours} h"); continue
            # a pod still loading the model is not idle: only count idleness after the first 45 min
            if up_h > 0.75 and util < a.idle_util:
                idle[p["id"]] += 1
                log(f"    idle {idle[p['id']]}/{a.idle_polls}")
                if idle[p["id"]] >= a.idle_polls:
                    terminate(p["id"], key, f"GPU under {a.idle_util}% for {a.idle_polls} polls")
            else:
                idle[p["id"]] = 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
