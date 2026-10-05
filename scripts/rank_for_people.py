#!/usr/bin/env python3
"""Order weak-label clips by how much a person's ear is needed, for the crowd site's limited storage
(backend/sync/working_set.py --fill reads the output in order until the storage budget is used).

Tiers, each interleaved across title classes so every sound gets coverage:
  1. contested   Gemini does not hear the sound the video's title names (the weak label is in doubt)
  2. unmapped    the title names a sound outside the label menu (writing, hand movements): only people can say
  3. agreed      Gemini hears the titled sound: two independent signals already agree, so people come last

  python rank_for_people.py --pool D:/t2a/pool_ytdense_all --out D:/t2a/pool_ytdense_all/site_manifest.jsonl
"""
from __future__ import annotations
import argparse, glob, json, re
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

MAP = {"typing": "tapping", "page turning": "paper rustling", "mouth sounds": "oral sounds"}
MENU = {"breathing", "kissing", "oral sounds", "moaning", "whispering", "normal speech", "tapping", "scratching", "crinkling",
        "brushing", "liquid", "spraying", "microphone touching", "sticky", "fabric rustling", "paper rustling", "cutting",
        "background music", "silence / room tone", "something else"}
UNCERTAINTY = {1: 0.95, 2: 0.9, 3: 0.6}


def safe(uid: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--judge", default="gemini-3.1-pro")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = {Path(f).stem: f for f in glob.glob(str(a.pool / "pool_*" / "clips" / "*.mp3"))}
    heard = {}
    for l in open(a.pool / f"{a.judge}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if not str(r.get("raw", "")).startswith("ERROR"): heard[r.get("clip") or r.get("uid")] = set(r["labels"])
    tiers: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for l in open(a.pool / "pool.jsonl", encoding="utf-8"):
        r = json.loads(l)
        f = files.get(safe(r["uid"]))
        if not f or Path(f).stat().st_size < 8000 or r["uid"] not in heard: continue
        title = MAP.get(r["label"], r["label"])
        tier = 2 if title not in MENU else (3 if title in heard[r["uid"]] else 1)
        tiers[tier][r["label"]].append({"uid": r["uid"], "path": f, "kind": "unlabelled", "uncertainty": UNCERTAINTY[tier],
                                        "info": {"title_class": r["label"], "src": "yt_dense", "gemini": sorted(heard[r["uid"]]),
                                                 "tier": tier}})
    rows = []
    for t in (1, 2, 3):
        for group in zip_longest(*tiers[t].values()):           # one of each class in turn
            rows += [x for x in group if x]
        print(f"tier {t}: {sum(map(len, tiers[t].values()))} clips " + str({c: len(v) for c, v in tiers[t].items()}))
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} clips ranked -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
