#!/usr/bin/env python3
"""Recordings whose titles put minors, school / teen settings or age play in a sexual scene. Nothing from them is
served to labellers or trained on: the fusion (merge_votes.py --exclude), CLAP training (train_clap_v8.py --exclude),
the crowd queue (sync/push_clips.py --exclude; sync clips already queued are blocked with admin_block_clips) all
read the uid list this writes.

Titles are the only screen here: ~49k recordings have hash file names and need their metadata screened separately.
Broad words that are usually innocent in these titles ("student", "young", "son") are not on the list, and neither is
adult family roleplay (siblings, cousins, step-family): only minor-coded terms are, parent-child incest among them.

  python content_filter.py --corpora aoxo/t2a-mommy aoxo/t2a-daddy --pools D:/t2a/pool* --out D:/t2a/fused/blocked
Out: <out>_recordings.txt (corpus file paths) and <out>_uids.txt (every pool uid from those recordings)
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path

BLOCK = re.compile(r"daughter|(step|mom|mommy|mum|mummy|mother|dad|daddy|father) ?sons?(?![a-z])|teen|school|classroom"
                   # tags often run together ("AgeDdlgIncest"), so the age-play acronyms match inside words too
                   r"|ddlg|ddlb|mdlg|mdlb|mmlg|mmlb|abdl|(?<![a-z])cgl(?![a-z])|little (one|girl|boy|princess)|age[- ]?play|ageplay"
                   r"|babysit|pubert|pubesc"
                   # the "[Age]" tag (age play / an age left open), but not "[No Age]", "age gap", "college age", "age 18"
                   r"|(?<![a-z])(?<!no )(?<!college )(?<!middle )(?<!legal )(?<!same )(?<!of )(?<!the )(?<!your )(?<!my )(?<!her )"
                   r"(?<!his )(?<!old )(?<!any )age(?![a-z])(?! gap| difference| appropriate| verified| legal| of | 1[89]| [2-9]\d)"
                   r"|(?<![a-z])(child|kid|kids|minor|underage|loli)(?![a-z])"
                   # 2026-10-22: school settings named without "school" ("Omega classmate in heat")
                   r"|classmates?|homeroom|(?<![a-z])prom(?![a-z])")
# 2026-11-01 (user): minor-coded only. Incest between adult siblings, cousins, aunts / uncles and step-siblings is no longer
# blocked (added 2026-10-18), nor are step-parents on their own. Parent-child incest still is, like "daughter" / "son":
# a title that says incest and names a parent ("Daddy adopts baby girl ... incest", "Mommy's Toy Incest").
PARENT_INCEST = re.compile(r"^(?=.*incest)(?=.*(dadd?y|(?<![a-z])dad(?![a-z])|father|papa|mommy|(?<![a-z])mom(?![a-z])"
                           r"|(?<![a-z])mum|mother|baby ?(girl|boy)))")


def blocked_title(path: str) -> bool:
    """True when a recording's file name (its title) names a blocked theme."""
    # titles separate words with hyphens, underscores or dots ("Little-Girl"): read them as spaces, or the multi-word
    # terms ("little girl", "big brother") never match (they did not before 2026-10-18)
    t = re.sub(r"[-_.]+", " ", path.split("/")[-1].lower())
    return bool(BLOCK.search(t) or PARENT_INCEST.search(t))


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
