#!/usr/bin/env python3
"""Mine the corpora we already own much harder: dense, overlapping gap windows.

The first pass cut each inter-word gap into *non-overlapping* windows of >= 1.5 s, so a 12 s pause became
one clip and a 1.2 s breath became nothing.  ASMR pacing means most of the non-speech audio lives in short
gaps, and a long gap usually contains several distinct events.  This pass re-reads the alignments already
on the Hub and emits:

  * windows from gaps as short as --min-gap (default 0.8 s) -- the label rule is <= 3-4 s, so short is fine
  * overlapping windows at --stride (default 3 s) inside longer gaps, instead of one window per span

Every uid already labeled or already pre-staged is skipped, so this only ever adds new work.  Output is
appended to labels/pending_candidates_dense.jsonl in each corpus repo, in the same schema the labeler and
prep stages consume.  Network-only: safe for the DO droplet, which holds nothing but the JSON it is reading.

  python3 mine_gaps_dense.py --repo aoxo/t2a-mommy --state /root/t2a/mine --workers 4
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import HfApi, hf_hub_download
from text2asmr.data.segment import load_alignment

PENDING_DENSE = "labels/pending_candidates_dense.jsonl"


def log(m): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def gaps(entries: list[dict]) -> list[tuple[float, float]]:
    """Silence spans between spoken words, from the alignment's own word timings."""
    words = [e for e in entries if (e.get("word") or e.get("text") or "").strip()]
    out = []
    prev_end = None
    for e in words:
        s, t = float(e.get("start", 0.0)), float(e.get("end", 0.0))
        if prev_end is not None and s > prev_end: out.append((prev_end, s))
        prev_end = max(prev_end or 0.0, t)
    return out


def windows(g0: float, g1: float, min_gap: float, win: float, stride: float, pad: float):
    """Overlapping windows covering one gap; the first window keeps the original pre-roll."""
    if g1 - g0 < min_gap: return
    cursor = g0
    while cursor < g1 - min_gap * 0.5:
        start = max(0.0, cursor - pad)
        dur = min(win, (g1 - cursor) + 2 * pad)
        yield cursor, round(start, 3), round(dur, 3)
        cursor += stride


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--state", type=Path, default=Path("/root/t2a/mine"))
    ap.add_argument("--min-gap", type=float, default=0.8)
    ap.add_argument("--window", type=float, default=8.0)
    ap.add_argument("--stride", type=float, default=3.0)
    ap.add_argument("--pad", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4, help="keep low: the droplet has 1 vCPU and the Hub rate-limits")
    ap.add_argument("--upload-every", type=int, default=2000, help="sources between Hub uploads")
    ap.add_argument("--max-sources", type=int, default=0)
    a = ap.parse_args()
    a.state.mkdir(parents=True, exist_ok=True)
    cache = a.state / "cache"; cache.mkdir(exist_ok=True)
    tag = a.repo.split("/")[-1]
    out_path = a.state / f"dense_{tag}.jsonl"
    done_path = a.state / f"done_{tag}.txt"
    api = HfApi()

    # everything already labeled or already queued: never re-emit it
    known: set[str] = set()
    for f in api.list_repo_files(a.repo, repo_type="dataset"):
        if f.startswith("labels/qwen3omni") and f.endswith(".jsonl") or f.startswith("labels/pending_candidates"):
            try: p = hf_hub_download(a.repo, f, repo_type="dataset", cache_dir=str(cache), force_download=True)
            except Exception as e: log(f"  skip {f} ({type(e).__name__})"); continue
            n = 0
            for line in open(p):
                try: known.add(json.loads(line)["uid"]); n += 1
                except Exception: pass
            log(f"  {f}: {n} uids (known {len(known)})")
            try: os.remove(os.path.realpath(p))
            except OSError: pass

    files = api.list_repo_files(a.repo, repo_type="dataset")
    js = {f[:-5] for f in files if f.endswith(".json")}
    sources = sorted(f for f in files if f.endswith(".m4a") and f in js)
    done = set(done_path.read_text().split("\n")) if done_path.exists() else set()
    todo = [s for s in sources if s not in done]
    if a.max_sources: todo = todo[: a.max_sources]
    log(f"{a.repo}: {len(sources)} transcribed sources, {len(done)} mined before, {len(todo)} to do")

    lock = threading.Lock(); stats = {"src": 0, "clips": 0, "skipped": 0}
    fh = out_path.open("a"); df = done_path.open("a")

    def one(src: str):
        try: entries = load_alignment(hf_hub_download(a.repo, src + ".json", repo_type="dataset", cache_dir=str(cache)))
        except Exception: return src, []
        rows = []
        for g0, g1 in gaps(entries):
            for gstart, cut_start, cut_dur in windows(g0, g1, a.min_gap, a.window, a.stride, a.pad):
                uid = f"{src}_{int(gstart * 1000):09d}"
                if uid in known: stats["skipped"] += 1; continue
                rows.append({"uid": uid, "source": src, "repo": a.repo, "start": round(gstart, 3),
                             "duration": round(min(g1 - gstart, a.window), 3), "old": None,
                             "cut_start": cut_start, "cut_duration": cut_dur, "dense": True})
        return src, rows

    def publish():
        if not out_path.exists() or out_path.stat().st_size == 0: return
        for attempt in range(4):
            try:
                api.upload_file(path_or_fileobj=str(out_path), path_in_repo=PENDING_DENSE, repo_id=a.repo,
                                repo_type="dataset", commit_message=f"dense gap mining: {stats['clips']} candidate clips")
                log(f"  uploaded {PENDING_DENSE} ({stats['clips']} clips)"); return
            except Exception as e:
                log(f"  upload retry {attempt}: {type(e).__name__} {str(e)[:110]}"); time.sleep(30 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for src, rows in ex.map(one, todo):
            with lock:
                for r in rows: fh.write(json.dumps(r) + "\n")
                stats["src"] += 1; stats["clips"] += len(rows)
                df.write(src + "\n")
                if stats["src"] % 200 == 0:
                    fh.flush(); df.flush()
                    log(f"  {stats['src']}/{len(todo)} sources, {stats['clips']} new clips, {stats['skipped']} already known")
                if stats["src"] % a.upload_every == 0:
                    fh.flush(); publish()
    fh.flush(); df.flush(); publish()
    log(f"MINE_DONE {a.repo} sources={stats['src']} new_clips={stats['clips']} already_known={stats['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
