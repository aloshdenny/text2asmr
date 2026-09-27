#!/usr/bin/env python3
"""Merge discovery outputs into the droplet's download queue, interleaving classes.

Discovery agents write one JSONL per class group. Rows are validated, deduplicated against each other and
against what is already downloaded, capped per channel, and emitted round-robin across classes so a
12-videos-per-hour fetcher fills every class at once instead of one class per day.

  python3 merge_candidates.py discover/group_*.jsonl --have have_ids.txt --out yt_candidates.jsonl
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

NEED = ("id", "cls", "kind")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--have", type=Path, help="ids already downloaded or queued")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-channel", type=int, default=5)
    a = ap.parse_args()
    have = set(a.have.read_text().split()) if a.have and a.have.exists() else set()
    by_cls, seen, chan, bad = defaultdict(list), set(), Counter(), Counter()
    for p in a.inputs:
        for line in p.open():
            try: r = json.loads(line)
            except Exception: bad["json"] += 1; continue
            if any(k not in r for k in NEED) or not re.fullmatch(r"[\w-]{11}", str(r["id"])): bad["schema"] += 1; continue
            if r["kind"] not in ("single", "chapter"): bad["kind"] += 1; continue
            if r["kind"] == "chapter":
                r["chapters"] = [c for c in r.get("chapters") or []
                                 if isinstance(c.get("start"), (int, float)) and isinstance(c.get("end"), (int, float))
                                 and c["end"] - c["start"] >= 60]
                if not r["chapters"]: bad["no chapters"] += 1; continue
            if r["id"] in seen or r["id"] in have: bad["dup"] += 1; continue
            key = (r["cls"], r.get("channel_id") or r.get("channel"))
            if chan[key] >= a.per_channel: bad["channel cap"] += 1; continue
            seen.add(r["id"]); chan[key] += 1
            r.setdefault("url", f"https://www.youtube.com/watch?v={r['id']}")
            by_cls[r["cls"]].append(r)
    rows = [r for group in zip_longest(*by_cls.values()) for r in group if r]
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} candidates -> {a.out}; per class " + ", ".join(f"{c}={len(v)}" for c, v in sorted(by_cls.items())))
    print("dropped: " + (", ".join(f"{k}={v}" for k, v in bad.items()) or "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
