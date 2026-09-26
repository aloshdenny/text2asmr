#!/usr/bin/env python3
"""Keep the data-collection pipeline full without a human in the loop.

The failure mode this exists for: a labeling pod runs its 8 h budget, stops cleanly, and then nothing
labels anything until someone notices. That happened overnight -- both pods exited and the GPUs sat unused
with $100 of credit available.

Every cycle it:
  * counts live t2a labeling pods and launches replacements, alternating corpora, up to --pods
  * refuses to launch below --balance-floor, and tries GPU types in order when one is supply-constrained
  * records a status snapshot (pods, balance, label counts, YouTube queue) to pipeline_status.json

It deliberately does *not* police idle or runaway pods -- runpod_guard.py owns that, and two processes
terminating the same pod would race. This one only ever starts work; the guard only ever stops it.

  python3 pipeline_supervisor.py --pods 2 --balance-floor 40 --interval 600
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, time, urllib.request
from pathlib import Path

API = "https://api.runpod.io/graphql"
GPUS = ["A100 PCIe", "A100 SXM", "RTX PRO 6000", "H100 PCIe"]   # cheapest first; skip what is unavailable


def log(m): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def gql(query: str, key: str) -> dict:
    req = urllib.request.Request(f"{API}?api_key={key}", data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "t2a-sup/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    if "errors" in d: raise RuntimeError(str(d["errors"])[:300])
    return d["data"]


def label_counts(cache: str) -> dict:
    """Per-class totals across both corpora, cheap enough to refresh on a slow cycle."""
    from collections import Counter
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); c: Counter = Counter()
    for repo in ("aoxo/t2a-mommy", "aoxo/t2a-daddy"):
        for f in api.list_repo_files(repo, repo_type="dataset"):
            if not (f.startswith("labels/qwen3omni") and f.endswith(".jsonl")): continue
            try: p = hf_hub_download(repo, f, repo_type="dataset", cache_dir=cache, force_download=True)
            except Exception: continue
            with open(p) as fh:
                for line in fh:
                    try: c[json.loads(line)["label"]] += 1
                    except Exception: pass
            try: os.remove(os.path.realpath(p))
            except OSError: pass
    return dict(c)


def launch(repo_key: str, budget_h: float, repo_dir: Path, python: str) -> bool:
    for gpu in GPUS:
        r = subprocess.run([python, str(repo_dir / "launch_label_pod.py"), "--repo-key", repo_key,
                            "--budget-h", str(budget_h), "--gpu", gpu, "--max-price", "2.2"],
                           capture_output=True, text=True, timeout=600)
        out = (r.stdout or "") + (r.stderr or "")
        if "launched" in out:
            log(f"  launched {repo_key} on {gpu}: {out.strip().splitlines()[-2][:90]}")
            return True
        reason = "supply" if "SUPPLY_CONSTRAINT" in out else out.strip().splitlines()[-1][:80] if out.strip() else "?"
        log(f"  {gpu} unavailable for {repo_key} ({reason})")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pods", type=int, default=2, help="live labeling pods to maintain")
    ap.add_argument("--budget-h", type=float, default=8.0)
    ap.add_argument("--balance-floor", type=float, default=40.0, help="never launch below this RunPod balance")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--repo-dir", type=Path, default=Path("/root/t2a"))
    ap.add_argument("--python", default="/root/t2a/venv/bin/python")
    ap.add_argument("--status", type=Path, default=Path("/root/t2a/pipeline_status.json"))
    ap.add_argument("--counts-every", type=int, default=6, help="cycles between label-count refreshes")
    a = ap.parse_args()
    key = os.environ["RUNPOD_API_KEY"]
    cycle = 0; counts: dict = {}
    next_side = "mommy"

    while True:
        cycle += 1
        try:
            m = gql("{ myself { clientBalance pods { id name desiredStatus costPerHr } } }", key)["myself"]
        except Exception as e:
            log(f"runpod poll failed: {type(e).__name__} {str(e)[:120]}"); time.sleep(a.interval); continue
        pods = [p for p in m["pods"] if (p.get("name") or "").startswith("t2a-label")]
        live = [p for p in pods if (p.get("desiredStatus") or "") == "RUNNING"]
        bal = m.get("clientBalance") or 0.0
        log(f"balance ${bal:.2f} | live labeling pods {len(live)}/{a.pods}")

        missing = a.pods - len(live)
        if missing > 0 and bal > a.balance_floor:
            for _ in range(missing):
                side = next_side
                next_side = "daddy" if side == "mommy" else "mommy"
                if not launch(side, a.budget_h, a.repo_dir, a.python):
                    log("  no GPU type available right now; will retry next cycle"); break
                time.sleep(20)
        elif missing > 0:
            log(f"  not launching: balance ${bal:.2f} is at or below the ${a.balance_floor} floor")

        if cycle % a.counts_every == 1:
            try: counts = label_counts(str(a.repo_dir / "supcache"))
            except Exception as e: log(f"  label counts failed: {type(e).__name__}")
        yt_done = yt_total = 0
        try:
            plog = Path("/root/t2a/pipeline.log").read_text(errors="replace")
            yt_done = plog.count("uploaded+verified")
            mt = re.findall(r"(\d+) urls, (\d+) already on HF", plog)
            if mt: yt_total = int(mt[-1][0])
        except Exception: pass
        snap = {"ts": time.strftime("%F %T"), "balance": bal, "live_pods": [p["name"] for p in live],
                "label_counts": counts, "yt_uploaded": yt_done, "yt_queue": yt_total}
        a.status.write_text(json.dumps(snap, indent=1))
        if counts:
            top = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:6])
            log(f"  labels: {top}")
        log(f"  youtube: {yt_done} uploaded of {yt_total} queued")
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
