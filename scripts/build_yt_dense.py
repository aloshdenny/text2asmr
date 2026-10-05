#!/usr/bin/env python3
"""Weak labels at scale from the YouTube trigger videos: windows across each verified video's labelled span, each
carrying the video's class (or its chapter's). Discovery picked the videos by title or chapters and the droplet
verified them before upload (VAD speech < 15% of the span, not silent), so a window's label is a weak vote about as
good as the title: tapping / crinkling ~73% precise, liquid / brushing ~45% (judge calibration). The CLAP probe and
a Gemini-checked sample per class sort out the rest.

Metadata comes from the droplet's candidate list (id, cls, kind, chapters, duration_s), not from per-file Hub reads:
the Hub allows 5,000 resolver requests per 5 minutes per account, and the window cutting needs them.

  python build_yt_dense.py --candidates yt_candidates.jsonl --done done.txt --rejects rejects.jsonl \\
      --pools D:/t2a/pool_yt D:/t2a/pool_mined ... --step 30 --per-video 80 --out D:/t2a/yt_dense/pool.jsonl
Rows: {uid "yt:<id>:<start>", label, src "yt_dense", repo, rec, start, kind}
"""
from __future__ import annotations
import argparse, json, random, re
from collections import Counter
from pathlib import Path

YT_REPO = "aoxo/asmr-yt-chapters"
WIN = 6.0                                   # what the labellers (and label_pool.py) hear


def spans(c: dict) -> list[tuple[float, float, str]]:
    """(start, end, class) spans worth windowing: a single-class video minus its first and last minute (intros and
    outros are where people talk), or each classified chapter minus 3 s at both ends."""
    dur = float(c.get("duration_s") or 0)
    if c.get("kind") != "chapter" or not c.get("chapters"):
        return [(60.0, dur - 60.0, c["cls"])] if dur > 180 else []
    out = []
    for ch in c["chapters"]:
        s = float(ch.get("start", ch.get("start_time", 0)) or 0); e = float(ch.get("end", ch.get("end_time", 0)) or 0)
        cls = ch.get("cls") or ch.get("label") or ch.get("class")
        if cls and e - s > WIN + 6: out.append((s + 3.0, e - 3.0, cls))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--done", type=Path, required=True, help="ids the droplet verified and uploaded")
    ap.add_argument("--rejects", type=Path, default=None)
    ap.add_argument("--pools", type=Path, nargs="*", default=[], help="pool dirs whose windows are already labelled")
    ap.add_argument("--step", type=float, default=30.0, help="seconds between window starts")
    ap.add_argument("--per-video", type=int, default=80, help="at most this many windows per video, spread evenly")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-per-class", type=int, default=0, help="also write a random sample of this many windows per class")
    ap.add_argument("--sample-out", type=Path, default=None, help="where the sample goes (a pool.jsonl for label_pool.py)")
    a = ap.parse_args()
    done = set(a.done.read_text().split())
    rejected = {json.loads(l)["id"] for l in open(a.rejects)} if a.rejects and a.rejects.exists() else set()
    seen = set()
    for d in a.pools:
        f = d / "pool.jsonl"
        if f.exists(): seen |= {json.loads(l)["uid"] for l in open(f, encoding="utf-8")}
    from huggingface_hub import HfApi                      # one listing call: which videos have audio on the Hub
    on_hub = {m.group(1) for f in HfApi().list_repo_files(YT_REPO, repo_type="dataset")
              if (m := re.match(r"audio/([A-Za-z0-9_-]{11})\.(m4a|webm|opus|mp3|flac|wav)$", f))}
    rows, per_cls, vids = [], Counter(), 0
    for l in open(a.candidates, encoding="utf-8"):
        c = json.loads(l)
        if c["id"] not in done or c["id"] in rejected or c["id"] not in on_hub: continue
        starts = [(t, cls) for s, e, cls in spans(c) for t in [s + k * a.step for k in range(int(max(0, e - s - WIN) // a.step) + 1)]]
        if not starts: continue
        if len(starts) > a.per_video:                        # spread the cap over the whole video
            starts = [starts[round(i * (len(starts) - 1) / (a.per_video - 1))] for i in range(a.per_video)]
        vids += 1
        for t, cls in starts:
            uid = f"yt:{c['id']}:{t:.1f}"
            if uid in seen: continue
            rows.append({"uid": uid, "label": cls, "src": "yt_dense", "repo": YT_REPO, "rec": c["id"], "start": round(t, 1),
                         "kind": c.get("kind", "single")})
            per_cls[cls] += 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} windows from {vids} videos -> {a.out}")
    for cls, n in per_cls.most_common(): print(f"  {cls:22} {n}")
    if a.sample_per_class and a.sample_out:
        # per class, at most 3 windows from one video, so the sample measures the class's titles, not one channel
        rng, sample = random.Random(0), []
        for cls in per_cls:
            pool = [r for r in rows if r["label"] == cls]; rng.shuffle(pool)
            per_vid, picked = Counter(), []
            for r in pool:
                if per_vid[r["rec"]] >= 3: continue
                per_vid[r["rec"]] += 1; picked.append(r)
                if len(picked) >= a.sample_per_class: break
            sample += picked
        a.sample_out.parent.mkdir(parents=True, exist_ok=True)
        a.sample_out.write_text("".join(json.dumps(r) + "\n" for r in sample), encoding="utf-8")
        print(f"sample: {len(sample)} windows ({a.sample_per_class} per class max) -> {a.sample_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
