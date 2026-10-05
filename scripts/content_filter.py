#!/usr/bin/env python3
"""Recordings whose titles put minors, school / teen settings, age play or incest in a sexual scene. Nothing from them is
served to labellers or trained on: the fusion (merge_votes.py --exclude), CLAP training (train_clap_v8.py --exclude),
the crowd queue (sync/push_clips.py --exclude; sync clips already queued are blocked with admin_block_clips) all
read the uid list this writes.

Titles are the only screen here: ~49k recordings have hash file names and need their metadata screened separately.
Broad words that are usually innocent in these titles ("brother", "student", "young", "son") are not on the list.

  python content_filter.py --corpora aoxo/t2a-mommy aoxo/t2a-daddy --pools D:/t2a/pool* --out D:/t2a/fused/blocked
Out: <out>_recordings.txt (corpus file paths) and <out>_uids.txt (every pool uid from those recordings)
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path

BLOCK = re.compile(r"daughter|step[- ]?(dad|mom|mum|daughter|son|sis|bro|father|mother)|teen|school|classroom"
                   r"|(?<![a-z])(ddlg|cgl|mdlb|ddlb|abdl)(?![a-z])|little (one|girl|boy)|age[- ]?play|ageplay|babysit"
                   r"|(?<![a-z])(child|kid|kids|minor|underage|loli)(?![a-z])")


def blocked_title(path: str) -> bool:
    """True when a recording's file name (its title) names a blocked theme."""
    return bool(BLOCK.search(path.split("/")[-1].lower().replace("_", " ")))


def rec_of(uid: str) -> str:
    return "yt:" + uid.split(":")[1] if uid.startswith("yt:") else re.sub(r"_\d+$", "", uid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=["aoxo/t2a-mommy", "aoxo/t2a-daddy"])
    ap.add_argument("--pools", type=Path, nargs="*", default=[])
    ap.add_argument("--transcripts", type=Path, nargs="*", default=[], help="screen_transcripts.py results: their blocked recordings join the list")
    ap.add_argument("--out", type=Path, required=True, help="prefix for <out>_recordings.txt and <out>_uids.txt")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    recs = sorted({f for repo in a.corpora for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.endswith((".m4a", ".mp3")) and blocked_title(f)})
    for f in a.transcripts:
        recs = sorted(set(recs) | {r["rec"] for r in map(json.loads, open(f, encoding="utf-8")) if r.get("blocked")})
    blocked = set(recs)
    uids = set()
    for d in a.pools:
        p = d / "pool.jsonl"
        if p.exists():
            uids |= {u for u in (json.loads(l)["uid"] for l in open(p, encoding="utf-8")) if rec_of(u) in blocked}
    Path(f"{a.out}_recordings.txt").write_text("".join(r + "\n" for r in recs), encoding="utf-8")
    Path(f"{a.out}_uids.txt").write_text("".join(u + "\n" for u in sorted(uids)), encoding="utf-8")
    print(f"{len(recs)} blocked recordings; {len(uids)} pool windows from them -> {a.out}_uids.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
