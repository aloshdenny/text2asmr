#!/usr/bin/env python3
"""Cut a window list (build_yt_dense.py) into the 6 s mp3 clips label_pool.py and pool_local_labels.py read from disk,
downloading each Hub file once instead of seeking it over HTTP per window (~50 windows per 250 MB YouTube FLAC).
Clips land in <out>/pool_dl/clips/<uid with unsafe characters replaced>.mp3 -- label_pool.py's own naming, so it
labels them without cutting again. Rows listed in --skip (already labelled elsewhere) are left out of <out>/pool.jsonl.

  python cut_dense_clips.py --windows D:/t2a/yt_dense/windows.jsonl --skip D:/t2a/pool_ytdense/pool.jsonl --out D:/t2a/pool_ytdense_all
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

YT_REPO = "aoxo/asmr-yt-chapters"


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def safe(uid: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def yt_paths() -> dict[str, str]:
    from huggingface_hub import HfApi
    return {m.group(1): f for f in HfApi().list_repo_files(YT_REPO, repo_type="dataset")
            if (m := re.match(r"audio/([A-Za-z0-9_-]{11})\.(m4a|webm|opus|mp3|flac|wav)$", f))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--skip", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    skip = {json.loads(l)["uid"] for f in a.skip for l in open(f, encoding="utf-8")}
    rows = [r for r in map(json.loads, open(a.windows, encoding="utf-8")) if r["uid"] not in skip]
    clips = a.out / "pool_dl" / "clips"; clips.mkdir(parents=True, exist_ok=True)
    (a.out / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    paths = yt_paths()
    by: dict[str, list] = {}
    for r in rows:
        if not (clips / f"{safe(r['uid'])}.mp3").exists(): by.setdefault(r["rec"], []).append(r)
    tmp = a.out / "tmp_dl"; tmp.mkdir(exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    log(f"{len(rows)} windows ({len(skip)} skipped); {sum(map(len, by.values()))} to cut from {len(by)} videos")

    def one(vid: str) -> int:
        import requests
        path = paths.get(vid)
        if not path: return 0
        local = tmp / f"{vid}{Path(path).suffix}"
        for attempt in range(6):
            try:
                with requests.get(f"https://huggingface.co/datasets/{YT_REPO}/resolve/main/{quote(path)}",
                                  headers={"Authorization": f"Bearer {token}"} if token else {}, stream=True, timeout=120) as r:
                    if r.status_code == 429: time.sleep(30 * (attempt + 1)); continue
                    r.raise_for_status()
                    with open(local, "wb") as fh:
                        for chunk in r.iter_content(1 << 20): fh.write(chunk)
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            return 0
        n = 0
        for w in by[vid]:
            dst = clips / f"{safe(w['uid'])}.mp3"
            p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{float(w['start']):.2f}", "-t", "6", "-i", str(local),
                                "-ac", "1", "-ar", "24000", "-b:a", "48k", str(dst)], capture_output=True, timeout=180)
            if p.returncode == 0 and dst.exists() and dst.stat().st_size > 8000: n += 1
            else: dst.unlink(missing_ok=True)
        local.unlink(missing_ok=True)
        return n

    done, t0 = 0, time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, n in enumerate(ex.map(one, list(by)), 1):
            done += n
            if i % 50 == 0: log(f"  {i}/{len(by)} videos, {done} clips, {(time.time() - t0) / 60:.0f} min")
    log(f"CUT_DONE {done} clips -> {clips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
