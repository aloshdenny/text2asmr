#!/usr/bin/env python3
"""Full-corpus CLAP pass: one 10 s window from every recording on the Hub that the CLAP index never covered, so the
probe (probe_fused.py, set "corpus") can label it. The index covers ~24k of ~126k recordings; this reaches the rest.

Each window is cut straight from the Hub copy (ffmpeg seeks over HTTP; no audio touches the disk) at --at seconds,
or from the start when the recording is shorter, and embedded with CLAP v7 like every index row. Parts are saved
every --part recordings and a rerun skips finished parts. HTTP 429 from the Hub backs off instead of failing.

  python embed_corpus_clap.py --covered D:/t2a/probe/rows.jsonl D:/t2a/pool/pool.jsonl ... --out D:/t2a/emb
  python embed_corpus_clap.py --windows D:/t2a/yt_dense/pool.jsonl --name ytdense --dur 6 --out D:/t2a/emb
Out: <out>/<name>.f16 (N x 512 float16, L2-normalised) + <name>.jsonl (uid, repo, rec, start, label)

--windows embeds exactly the listed windows (rows with uid, repo, rec, start, as build_yt_dense.py writes) instead of
one window per uncovered recording. --download fetches each file once and cuts all its windows locally (then deletes
it): with ~50 windows per 250 MB YouTube FLAC, seeking over HTTP per window cost many range requests each and ran
into the Hub's 5,000-requests-per-5-minutes limit.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_pool_clap import SR, Scorer, repeatpad

CKPT_REPO = "aoxo/clap-ft-data"
YT_REPO = "aoxo/asmr-yt-chapters"
CORPORA = ("aoxo/t2a-mommy", "aoxo/t2a-daddy", "aoxo/t2a-audios-v1", YT_REPO)


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def rec_of(uid: str) -> str:
    return "yt:" + uid.split(":")[1] if uid.startswith("yt:") else re.sub(r"_\d+$", "", uid)


def recordings() -> list[tuple[str, str, str]]:
    """(repo, file path, recording id) for every recording the rings count: the two corpora's audio files,
    t2a-audios-v1's top-level m4a (not its pre-cut segments) and the YouTube chapter videos."""
    from huggingface_hub import HfApi
    out = []
    for repo in CORPORA:
        for f in HfApi().list_repo_files(repo, repo_type="dataset"):
            if repo == YT_REPO:
                m = re.match(r"audio/([A-Za-z0-9_-]{11})\.(m4a|webm|opus|mp3|flac|wav)$", f)
                if m: out.append((repo, f, f"yt:{m.group(1)}"))
            elif repo == "aoxo/t2a-audios-v1":
                if "/" not in f and f.endswith(".m4a"): out.append((repo, f, f))
            elif f.endswith((".m4a", ".mp3")):
                out.append((repo, f, f))
    return out


class Fetcher:
    def __init__(self, token: str | None):
        self.headers = ["-headers", f"Authorization: Bearer {token}\r\n"] if token else []
        self.lock = threading.Lock()
        self.pause_until = 0.0
        self.failed = 0
        self.limited = 0

    def window(self, repo: str, path: str, at: float, dur: float = 10.0, exact: bool = False) -> tuple[np.ndarray | None, float]:
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{quote(path)}"
        for start in ((at,) if exact else (at, 0.0)):
            for attempt in range(5):
                wait = self.pause_until - time.time()
                if wait > 0: time.sleep(wait)
                r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *self.headers, "-ss", f"{start:.1f}", "-t", f"{dur}",
                                    "-i", url, "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1", "-"],
                                   capture_output=True, timeout=240)
                err = r.stderr.decode(errors="ignore")
                if "429" in err:                           # rate limited: everyone waits, then this one retries
                    with self.lock:
                        self.limited += 1
                        self.pause_until = max(self.pause_until, time.time() + 30 * (attempt + 1))
                    continue
                y = np.frombuffer(r.stdout, dtype=np.float32)
                if y.size >= SR: return y.copy(), start        # at least 1 s of audio
                if r.returncode != 0 and attempt < 2 and not y.size and "Invalid data" not in err:
                    time.sleep(3 * (attempt + 1)); continue  # a network hiccup: retry this start
                break                                         # too short here: try from the start
        with self.lock: self.failed += 1
        return None, 0.0


def file_windows(fe: Fetcher, token: str | None, tmp: Path, repo: str, path: str, items: list, dur: float) -> list:
    """Download one Hub file, cut every listed window from the local copy, delete it. [(row, wav or None)]"""
    import requests
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{quote(path)}"
    local = tmp / f"{abs(hash((repo, path)))}{Path(path).suffix}"
    for attempt in range(6):
        wait = fe.pause_until - time.time()
        if wait > 0: time.sleep(wait)
        try:
            with requests.get(url, headers={"Authorization": f"Bearer {token}"} if token else {}, stream=True, timeout=120) as r:
                if r.status_code == 429:
                    with fe.lock:
                        fe.limited += 1; fe.pause_until = max(fe.pause_until, time.time() + 30 * (attempt + 1))
                    continue
                r.raise_for_status()
                with open(local, "wb") as fh:
                    for chunk in r.iter_content(1 << 20): fh.write(chunk)
            break
        except Exception:
            time.sleep(5 * (attempt + 1))
    else:
        with fe.lock: fe.failed += len(items)
        return [(it, None) for it in items]
    out = []
    for it in items:
        r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{float(it['start']):.2f}", "-t", f"{dur}", "-i", str(local),
                            "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1", "-"], capture_output=True, timeout=180)
        y = np.frombuffer(r.stdout, dtype=np.float32)
        out.append((it, y.copy() if y.size >= SR else None))
    local.unlink(missing_ok=True)
    return out


def by_file(a, todo: list, sc: Scorer, fe: Fetcher, parts: Path) -> int:
    """--windows --download: group the windows by file, a few files at a time."""
    groups: dict[tuple[str, str], list] = {}
    for repo, path, rec, start, row in todo:
        groups.setdefault((repo, path), []).append({**row, "rec": rec, "start": start})
    keys = list(groups); tmp = a.out / "tmp_dl"; tmp.mkdir(parents=True, exist_ok=True)
    log(f"{len(todo)} windows in {len(keys)} files")
    token, t0 = os.environ.get("HF_TOKEN"), time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for k in range(0, len(keys), a.part_files):
            name = parts / f"part_{k // a.part_files:04d}"
            if Path(f"{name}.jsonl").exists(): continue
            rows, E, buf = [], [], []

            def flush():
                E.append(sc.embed_mels(sc.mels(np.stack([repeatpad(w) for _, w in buf]))).half().cpu().numpy())
                rows.extend(r for r, _ in buf); buf.clear()

            chunk = keys[k:k + a.part_files]
            for res in ex.map(lambda key: file_windows(fe, token, tmp, key[0], key[1], groups[key], a.dur), chunk):
                for it, y in res:
                    if y is None:
                        with fe.lock: fe.failed += 1
                        continue
                    buf.append(({"uid": it["uid"], "repo": it["repo"], "rec": it["rec"], "start": it["start"],
                                 "label": it.get("label"), "src": it.get("src")}, y))
                    if len(buf) >= a.batch: flush()
            if buf: flush()
            if E: np.concatenate(E).tofile(f"{name}.f16")
            Path(f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            log(f"  {min(k + a.part_files, len(keys))}/{len(keys)} files, {len(rows)} windows in this part, {fe.failed} unreadable, "
                f"{fe.limited} rate-limit waits, {(time.time() - t0) / 60:.0f} min")
    rows, E = [], []
    for f in sorted(parts.glob("part_*.jsonl")):
        rs = [json.loads(l) for l in open(f, encoding="utf-8")]
        if not rs: continue
        rows += rs; E.append(np.fromfile(f.with_suffix(".f16"), dtype=np.float16).reshape(len(rs), -1))
    np.concatenate(E).tofile(a.out / f"{a.name}.f16")
    (a.out / f"{a.name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    log(f"CORPUS_EMBED_DONE {len(rows)} windows -> {a.out / (a.name + '.f16')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--covered", type=Path, nargs="*", default=[], help="jsonl files whose uids' recordings are already covered")
    ap.add_argument("--windows", type=Path, default=None, help="embed these windows (jsonl: uid, repo, rec, start) instead")
    ap.add_argument("--name", default="corpus", help="output set name: <out>/<name>.f16 / .jsonl")
    ap.add_argument("--dur", type=float, default=10.0, help="window length (s)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", default="v7_ckpt/stage2_best")
    ap.add_argument("--at", type=float, default=30.0, help="window start (s); shorter recordings use their start")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--part", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="only the first N uncovered recordings (a trial run)")
    ap.add_argument("--download", action="store_true", help="with --windows: download each file once, cut its windows locally")
    ap.add_argument("--part-files", type=int, default=20, help="with --download: files per saved part")
    a = ap.parse_args()
    if a.windows:                                            # explicit windows: (repo, path, rec, start, row)
        paths = {rec: path for repo, path, rec in recordings()}
        rows_in = [json.loads(l) for l in open(a.windows, encoding="utf-8")]
        todo = [(r["repo"], paths[k], k, float(r["start"]), r) for r in rows_in
                if (k := r["rec"] if str(r["rec"]).startswith("yt:") or r["repo"] != YT_REPO else f"yt:{r['rec']}") in paths]
        log(f"{len(todo)} of {len(rows_in)} windows have audio on the Hub")
    else:
        covered = set()
        for f in a.covered:
            for l in open(f, encoding="utf-8"): covered.add(rec_of(json.loads(l)["uid"]))
        todo = [(repo, path, rec, a.at, None) for repo, path, rec in sorted(r for r in recordings() if r[2] not in covered)]
        log(f"{len(covered)} recordings covered already; {len(todo)} to embed")
    if a.limit: todo = todo[:a.limit]
    parts = a.out / f"parts_{a.name}"; parts.mkdir(parents=True, exist_ok=True)
    ck = Path(a.ckpt)
    if not (ck / "config.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(CKPT_REPO, repo_type="dataset", allow_patterns=[f"{a.ckpt}/*"], local_dir=a.out / "ckpt")
        ck = a.out / "ckpt" / a.ckpt
    sc = Scorer(ck, dev="cuda")
    fe = Fetcher(os.environ.get("HF_TOKEN"))
    if a.windows and a.download:
        return by_file(a, todo, sc, fe, parts)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for k in range(0, len(todo), a.part):
            name = parts / f"part_{k // a.part:04d}"
            if Path(f"{name}.jsonl").exists(): continue
            chunk = todo[k:k + a.part]
            rows, E, buf = [], [], []

            def flush():
                E.append(sc.embed_mels(sc.mels(np.stack([repeatpad(w) for _, w in buf]))).half().cpu().numpy())
                rows.extend(r for r, _ in buf); buf.clear()

            fetch = lambda t: fe.window(t[0], t[1], t[3], a.dur, exact=t[4] is not None)
            for (repo, path, rec, _, src), (y, start) in zip(chunk, ex.map(fetch, chunk)):
                if y is None: continue
                if src is not None:                          # a listed window keeps its own uid and label
                    buf.append(({"uid": src["uid"], "repo": repo, "rec": rec, "start": start, "label": src.get("label"),
                                 "src": src.get("src")}, y))
                    if len(buf) >= a.batch: flush()
                    continue
                uid = f"{rec}:{start:.1f}" if rec.startswith("yt:") else f"{rec}_{int(start * 1000)}"
                buf.append(({"uid": uid, "repo": repo, "rec": rec, "start": start}, y))
                if len(buf) >= a.batch: flush()
            if buf: flush()
            if E: np.concatenate(E).tofile(f"{name}.f16")
            Path(f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            done = min(k + a.part, len(todo))
            log(f"  {done}/{len(todo)} recordings, {len(rows)} embedded in this part, {fe.failed} unreadable, "
                f"{fe.limited} rate-limit waits, {(time.time() - t0) / 60:.0f} min")
    # all parts -> one set, in part order
    rows, E = [], []
    for f in sorted(parts.glob("part_*.jsonl")):
        rs = [json.loads(l) for l in open(f, encoding="utf-8")]
        if not rs: continue
        rows += rs; E.append(np.fromfile(f.with_suffix(".f16"), dtype=np.float16).reshape(len(rs), -1))
    np.concatenate(E).tofile(a.out / f"{a.name}.f16")
    (a.out / f"{a.name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    log(f"CORPUS_EMBED_DONE {len(rows)} windows -> {a.out / (a.name + '.f16')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
