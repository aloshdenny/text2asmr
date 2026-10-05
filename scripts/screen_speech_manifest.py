#!/usr/bin/env python3
"""Content filter for the T2A generator's data (aoxo/t2a-speech-v2): every speech window that comes from a blocked
recording (content_filter.py's title screen, screen_transcripts.py's transcript screen) or whose own script names
minors, school / teen settings, age play or incest -- or a stated age under 18 -- goes on the exclusion list that
train_t3_v2.py, dpo_pairs_t3.py and build_voice_bank.py skip by default.

  python screen_speech_manifest.py --blocked-recordings D:/t2a/fused/blocked_recordings.txt \\
      --transcripts D:/t2a/fused/transcript_screen.jsonl --out D:/t2a/fused/blocked_speech_uids.txt
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_filter import blocked_title
from screen_transcripts import AGES, TERMS


def rec_of(uid: str) -> str: return re.sub(r"_\d+$", "", uid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-repo", default="aoxo/t2a-speech-v2")
    ap.add_argument("--blocked-recordings", type=Path, nargs="*", default=[])
    ap.add_argument("--transcripts", type=Path, nargs="*", default=[], help="screen_transcripts.py results")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    blocked = {l.strip() for f in a.blocked_recordings for l in open(f, encoding="utf-8") if l.strip()}
    for f in a.transcripts:
        blocked |= {r["rec"] for r in map(json.loads, open(f, encoding="utf-8")) if r.get("blocked")}
    man = hf_hub_download(a.manifest_repo, "manifests/speech_windows_v2.jsonl", repo_type="dataset")
    out, why, n = [], Counter(), 0
    for l in open(man, encoding="utf-8"):
        r = json.loads(l); n += 1
        rec = rec_of(r["uid"]); text = (r.get("text") or "").lower()
        if rec in blocked: out.append(r["uid"]); why["blocked recording"] += 1
        elif blocked_title(rec): out.append(r["uid"]); why["blocked title"] += 1
        elif TERMS.search(text) or AGES.search(text): out.append(r["uid"]); why["script"] += 1
    a.out.write_text("".join(u + "\n" for u in out), encoding="utf-8")
    print(f"{len(out)} of {n} speech windows excluded ({100 * len(out) / max(1, n):.1f}%): {dict(why)} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
