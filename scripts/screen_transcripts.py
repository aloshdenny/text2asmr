#!/usr/bin/env python3
"""Screen recordings whose file names carry no title (soundgasm media hashes) by what is said in them: each one's word
transcript (<file>.json on the Hub) is read and matched against unambiguous terms for minors, school / teen settings,
age play and incest, plus stated ages under 18. "mommy" / "daddy" alone do not count: in these corpora they are adult
kink terms. A recording with any match is blocked (content_filter.py's lists pick it up).

Transcripts are streamed and parsed in memory (nothing written but the results); Hub 429s back off. Resumable.

  python screen_transcripts.py --corpora aoxo/t2a-mommy aoxo/t2a-daddy --out D:/t2a/fused/transcript_screen.jsonl
"""
from __future__ import annotations
import argparse, json, os, re, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests

TERMS = re.compile(r"\b(daughter|step ?daughter|step ?son|step ?sister|step ?brother|step ?dad|step ?mom|step ?mum|little girl|little boy"
                   r"|high ?school|middle school|junior high|underage|loli|age ?play|babysitter|babysit|babysitting|ddlg|cgl|abdl)\b")
AGES = re.compile(r"\b(1[0-7]|[1-9]|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen)[ -]years?[ -]old\b")
HASHED = re.compile(r"[^/]+/[0-9a-f]{40}\.(m4a|mp3)$")


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def text_of(doc) -> str:
    words = doc if isinstance(doc, list) else doc.get("words") or doc.get("segments") or []
    return " ".join(w.get("word") or w.get("text") or "" for w in words if isinstance(w, dict)).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=["aoxo/t2a-mommy", "aoxo/t2a-daddy"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    from huggingface_hub import HfApi
    done = {json.loads(l)["rec"] for l in open(a.out, encoding="utf-8")} if a.out.exists() else set()
    todo = []
    for repo in a.corpora:
        fs = set(HfApi().list_repo_files(repo, repo_type="dataset"))
        todo += [(repo, f) for f in sorted(fs) if HASHED.fullmatch(f) and f + ".json" in fs and f not in done]
    log(f"{len(todo)} untitled recordings with transcripts to screen ({len(done)} done before)")
    token = os.environ.get("HF_TOKEN"); hdr = {"Authorization": f"Bearer {token}"} if token else {}
    lock, pause, stats = threading.Lock(), [0.0], Counter()

    def one(item):
        repo, f = item
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{quote(f + '.json')}"
        for attempt in range(6):
            wait = pause[0] - time.time()
            if wait > 0: time.sleep(wait)
            try:
                r = requests.get(url, headers=hdr, timeout=120)
                if r.status_code == 429:
                    with lock: pause[0] = max(pause[0], time.time() + 30 * (attempt + 1)); stats["429"] += 1
                    continue
                r.raise_for_status()
                t = text_of(r.json())
                hits = Counter(m.group(0) for m in TERMS.finditer(t)) + Counter(m.group(0) for m in AGES.finditer(t))
                return {"rec": f, "repo": repo, "blocked": bool(hits), "hits": dict(hits), "words": len(t.split())}
            except Exception:
                time.sleep(5 * (attempt + 1))
        return {"rec": f, "repo": repo, "blocked": None, "error": True}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a", encoding="utf-8") as fh:
        for i, r in enumerate(ex.map(one, todo), 1):
            fh.write(json.dumps(r) + "\n"); stats["blocked" if r.get("blocked") else "error" if r.get("error") else "clean"] += 1
            if i % 2000 == 0:
                fh.flush(); log(f"  {i}/{len(todo)} screened, {dict(stats)}, {(time.time() - t0) / 60:.0f} min")
    log(f"SCREEN_DONE {dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
