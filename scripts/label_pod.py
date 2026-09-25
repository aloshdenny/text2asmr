#!/usr/bin/env python3
"""Qwen3-Omni labeling on a RunPod pod: the Modal job, without Modal.

Consumes, in order of preference:
  1. labels/label_queue.jsonl        -- CLAP-ranked queue (rare classes first), when the scorer has run
  2. labels/pending_candidates*.jsonl -- the raw backlog, oldest pass first

Writes each chunk straight to the corpus ledger on the Hub, so a pod that dies (spot preemption, watchdog,
balance floor) loses at most one chunk and the next pod resumes from the ledger.  Prints LABEL_RUN_DONE when
the work or the budget runs out, which is the watchdog's signal to terminate the pod.

  python3 label_pod.py --repo-key mommy --budget-h 8 --shard 0 --n-shards 1
"""
from __future__ import annotations
import argparse, json, os, subprocess, time, zlib
from collections import defaultdict
from pathlib import Path

CORPUS = {"mommy": ("aoxo/t2a-mommy", "labels/qwen3omni_expansion.jsonl"),
          "daddy": ("aoxo/t2a-daddy", "labels/qwen3omni.jsonl")}
MODEL = os.environ.get("T2A_LABEL_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
REPO_DIR = Path(os.environ.get("T2A_DIR", "/workspace/t2a"))


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def serve() -> subprocess.Popen:
    import urllib.request
    srv = subprocess.Popen(["vllm", "serve", MODEL, "--dtype", "bfloat16", "--max-model-len", "4096",
                            "--limit-mm-per-prompt", '{"audio":1}', "--gpu-memory-utilization", "0.90",
                            "--port", "8000"],
                           stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(200):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5); log("vllm up"); return srv
        except Exception:
            if srv.poll() is not None:
                log(open("/tmp/vllm.log").read()[-4000:]); raise SystemExit("vllm died during startup")
            time.sleep(15)
    raise SystemExit("vllm did not come up in 50 min")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-key", default="mommy", choices=sorted(CORPUS))
    ap.add_argument("--budget-h", type=float, default=8.0, help="wall-clock cap; the pod stops cleanly at it")
    ap.add_argument("--chunk", type=int, default=40000)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--work", type=Path, default=Path("/workspace/lab"))
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    repo, ledger = CORPUS[a.repo_key]
    if a.n_shards > 1: ledger = ledger.replace(".jsonl", f".s{a.shard}.jsonl")
    api = HfApi(); a.work.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); budget_s = a.budget_h * 3600

    files = api.list_repo_files(repo, repo_type="dataset")
    done: set[str] = set()
    for f in files:
        if f.startswith("labels/qwen3omni") and f.endswith(".jsonl"):
            try:
                for line in open(hf_hub_download(repo, f, repo_type="dataset", force_download=True)):
                    done.add(json.loads(line)["uid"])
            except Exception as e: log(f"  ledger {f} unreadable ({type(e).__name__})")
    log(f"{len(done)} uids already labeled")

    pools = ["labels/label_queue.jsonl"] if "labels/label_queue.jsonl" in files else []
    pools += sorted(f for f in files if f.startswith("labels/pending_candidates"))
    todo = []
    seen = set()
    for name in pools:
        try: p = hf_hub_download(repo, name, repo_type="dataset", force_download=True)
        except Exception as e: log(f"  pool {name} unreadable ({type(e).__name__})"); continue
        n = 0
        for line in open(p):
            try: r = json.loads(line)
            except Exception: continue
            uid = r["uid"]
            if uid in done or uid in seen: continue
            seen.add(uid); todo.append(r); n += 1
        log(f"  {name}: +{n} unlabeled")
        if len(todo) > 4_000_000: break          # plenty queued; stop reading multi-GB pools
    if a.n_shards > 1:
        todo = [r for r in todo if zlib.crc32(r["uid"].encode()) % a.n_shards == a.shard]
    log(f"{repo}: {len(todo)} clips to label (shard {a.shard}/{a.n_shards})")
    if not todo: log("LABEL_RUN_DONE nothing to do"); return 0

    by_src: dict[str, list] = defaultdict(list)
    for r in todo: by_src[r["source"]].append(r)
    chunks, cur = [], []
    for src in by_src:                            # chunk on source boundaries: prep fetches per source
        cur.extend(by_src[src])
        if len(cur) >= a.chunk: chunks.append(cur); cur = []
    if cur: chunks.append(cur)
    log(f"{len(chunks)} chunks over {len(by_src)} sources")

    srv = serve(); total = 0
    for ci, part in enumerate(chunks):
        left = budget_s - (time.time() - t0)
        if left < 1800:
            log(f"budget spent ({(time.time()-t0)/3600:.1f} h); stopping cleanly"); break
        (a.work / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
        for f in ("labels.jsonl", "clap_index.jsonl", "prep_done_sources.txt", "prep.log"):
            (a.work / f).unlink(missing_ok=True)
        log(f"chunk {ci}/{len(chunks)}: {len(part)} clips, {left/3600:.1f} h left")
        prep = subprocess.Popen(["python", str(REPO_DIR / "scripts/label_audios2_qwen3.py"), "--stage", "prep",
                                 "--work", str(a.work), "--workers", "12", "--bg-max", "1.01", "--no-clap"],
                                cwd=REPO_DIR, stdout=open(a.work / "prep.log", "w"), stderr=subprocess.STDOUT)
        (a.work / "clap_index.jsonl").touch()
        subprocess.run(["python", str(REPO_DIR / "scripts/label_audios2_qwen3.py"), "--stage", "label",
                        "--work", str(a.work), "--concurrency", str(a.concurrency), "--delete-wav",
                        "--no-clap", "--follow", "--chunk", "4000"], cwd=REPO_DIR, check=True)
        try: prep.wait(timeout=900)
        except Exception: prep.kill()

        new = [json.loads(l) for l in open(a.work / "labels.jsonl")] if (a.work / "labels.jsonl").exists() else []
        idx = {r["uid"]: r for r in part}
        rows = [dict(r, source=idx[r["uid"]]["source"], repo=repo) for r in new if r["uid"] in idx]
        if not rows: log("chunk produced no rows"); continue
        old = []
        try: old = [json.loads(l) for l in open(hf_hub_download(repo, ledger, repo_type="dataset", force_download=True))]
        except Exception: pass
        out = a.work / "ledger.jsonl"; out.write_text("".join(json.dumps(r) + "\n" for r in old + rows))
        for attempt in range(5):
            try:
                api.upload_file(path_or_fileobj=str(out), path_in_repo=ledger, repo_id=repo, repo_type="dataset",
                                commit_message=f"+{len(rows)} Qwen3-Omni labels"); break
            except Exception as e:
                log(f"upload retry {attempt}: {type(e).__name__} {str(e)[:120]}"); time.sleep(30 * (attempt + 1))
        total += len(rows)
        log(f"uploaded {ledger}: +{len(rows)} (run total {total}, ledger {len(old)+len(rows)})")
    srv.kill()
    log(f"LABEL_RUN_DONE labeled={total} in {(time.time()-t0)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
