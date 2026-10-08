#!/usr/bin/env python3
"""Pick the clips people should hear next from a fused weak-label pool: the ones the judges' fusion is unsure about.

A clip is unsure when any class's fused P(present) sits between --lo and --hi; its uncertainty is the largest
1 - |2P - 1| over classes (what the site's need score ranks by). Confident clips stay off the site. Clips whose title
class or unsure classes are among --boost (the classes people have labelled least) come first; within each tier the
classes take turns, most uncertain first, so every sound gets heard. Output: working_set.py --fill manifest rows
{uid, path, kind "unsure", uncertainty, info}.

  python rank_fused_for_people.py --fused D:/t2a/fused/fused.jsonl --pool D:/t2a/pool_ytdense2 \\
      --boost spraying "paper rustling" cutting sticky brushing liquid --exclude D:/t2a/fused/blocked_uids.txt \\
      --out D:/t2a/pool_ytdense2/site_manifest.jsonl
"""
from __future__ import annotations
import argparse, glob, json, re
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

TITLE_MAP = {"typing": "tapping", "page turning": "paper rustling", "mouth sounds": "oral sounds"}


def safe(uid: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--pool", type=Path, nargs="+", required=True, help="pool dirs (pool.jsonl + pool_*/clips)")
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--boost", nargs="*", default=[])
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    drop = {u.strip() for f in a.exclude for u in open(f, encoding="utf-8") if u.strip()}
    fused = {}
    for l in open(a.fused, encoding="utf-8"):
        r = json.loads(l); fused[r["uid"]] = r["probs"]
    boost = set(a.boost)
    tiers: dict[int, dict[str, list]] = {0: defaultdict(list), 1: defaultdict(list)}
    why = Counter()
    for d in a.pool:
        files = {Path(f).stem: f for f in glob.glob(str(d / "pool_*" / "clips" / "*.mp3"))}
        for l in open(d / "pool.jsonl", encoding="utf-8"):
            r = json.loads(l); u = r["uid"]
            if u in drop: why["content filter"] += 1; continue
            f, P = files.get(safe(u)), fused.get(u)
            if not f: why["no audio"] += 1; continue
            if P is None: why["not fused"] += 1; continue
            unsure = sorted(c for c, p in P.items() if a.lo <= p <= a.hi)
            if not unsure: why["confident"] += 1; continue
            title = TITLE_MAP.get(r.get("label"), r.get("label"))
            unc = max(1 - abs(2 * p - 1) for p in P.values())
            tier = 0 if title in boost or boost & set(unsure) else 1
            tiers[tier][title].append({"uid": u, "path": f, "kind": "unsure", "uncertainty": round(unc, 4),
                                       "info": {"title_class": r.get("label"), "src": r.get("src", "yt_dense"), "unsure": unsure,
                                                "p": {c: round(p, 3) for c, p in P.items() if p >= 0.1}}})
            why[f"unsure, tier {tier}"] += 1
    rows = []
    for t in (0, 1):
        for v in tiers[t].values(): v.sort(key=lambda x: -x["uncertainty"])
        rows += [x for group in zip_longest(*tiers[t].values()) for x in group if x]
        print(f"tier {t}: " + ", ".join(f"{c} {len(v)}" for c, v in sorted(tiers[t].items(), key=lambda x: -len(x[1]))))
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} unsure clips -> {a.out}; {dict(why)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
