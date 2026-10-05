#!/usr/bin/env python3
"""List pool clips whose audio is unreadable or near-empty (a remote cut that came out as one mp3 frame). The LLM
judges still answered for them, so their votes describe no sound at all: merge_votes.py --exclude drops them.

  python find_broken_clips.py --pools D:/t2a/pool D:/t2a/pool_yt ... --out D:/t2a/fused/broken_clips.txt
"""
from __future__ import annotations
import argparse, glob, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_local_labels import try_load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    broken = []
    for pool in a.pools:
        files = {Path(f).stem: f for f in glob.glob(str(pool / "pool_*" / "clips" / "*.mp3"))}
        n = 0
        for l in open(pool / "pool.jsonl", encoding="utf-8"):
            uid = json.loads(l)["uid"]; f = files.get(re.sub(r"[^A-Za-z0-9_.-]", "_", uid))
            if f and try_load(f, 16000) is None: broken.append(uid); n += 1
        print(f"{pool}: {n} broken of {len(files)} clips on disk", flush=True)
    a.out.write_text("".join(u + "\n" for u in broken), encoding="utf-8")
    print(f"{len(broken)} broken clips -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
