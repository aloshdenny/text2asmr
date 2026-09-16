#!/usr/bin/env python3
"""Select a class-tempered, source-diverse CLAP v2 training subset + reject negatives."""
from __future__ import annotations
import argparse, glob, json, random, zlib
from collections import defaultdict
from pathlib import Path

REJECT_TEXTS = [
    "a person talking, normal speech, not an ASMR trigger sound",
    "spoken voice, conversation, no trigger",
    "silence or room tone, no ASMR trigger",
    "speech or moaning, not a close-mic trigger sound",
]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ledgers", default="label_tool/gemini_audios2*.jsonl")
    ap.add_argument("--out", default="label_tool/v2/train_subset.jsonl")
    ap.add_argument("--total", type=int, default=800_000)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--per-source-cap", type=int, default=60)
    ap.add_argument("--min-per-source", type=int, default=8)
    ap.add_argument("--rejects", type=int, default=60_000)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for line in open(a.manifest):
        if not line.strip(): continue
        r = json.loads(line)
        au = r["audio"]
        by_class[r["trigger"]].append({
            "uid": r["uid"], "label": r["trigger"], "text": r["text"],
            "source": au["source"], "start": au["start"], "duration": au["duration"],
        })
    seen = {r["uid"] for rows in by_class.values() for r in rows}
    rej = []
    for p in glob.glob(a.ledgers):
        for line in open(p):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("gemini_label") != "reject" or not r.get("duration") or r["uid"] in seen: continue
            seen.add(r["uid"])
            rej.append({"uid": r["uid"], "label": "reject", "text": REJECT_TEXTS,
                        "source": r["source"], "start": r["start"], "duration": r["duration"]})
    rng.shuffle(rej)
    by_class["reject"] = rej[: a.rejects]
    print(f"classes={len(by_class)} manifest_rows={len(seen)-len(rej)} rejects_avail={len(rej)}")

    # tempered class targets with iterative cap redistribution
    n = {c: len(v) for c, v in by_class.items()}
    target = {}
    remaining = a.total; open_c = set(n)
    while open_c:
        w = {c: n[c] ** a.alpha for c in open_c}; s = sum(w.values())
        capped = False
        for c in list(open_c):
            t = int(remaining * w[c] / s)
            if t >= n[c]:
                target[c] = n[c]; remaining -= n[c]; open_c.discard(c); capped = True
        if not capped:
            for c in open_c: target[c] = int(remaining * w[c] / s)
            break
    # per-source cap: shuffle then take
    sel = []
    for c, rows in by_class.items():
        rng.shuffle(rows); per_src = defaultdict(int); k = 0
        for r in rows:
            if k >= target[c]: break
            if per_src[r["source"]] >= a.per_source_cap: continue
            per_src[r["source"]] += 1; sel.append(r); k += 1
    # drop sources that contribute too few clips (each costs a ~16MB download)
    src_n = defaultdict(int)
    for r in sel: src_n[r["source"]] += 1
    keep_src = {s for s, k in src_n.items() if k >= a.min_per_source}
    sel = [r for r in sel if r["source"] in keep_src]
    # source-level eval split
    for r in sel:
        r["split"] = "eval" if (zlib.crc32(r["source"].encode()) % 1000) < a.eval_frac * 1000 else "train"
    rng.shuffle(sel)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in sel: f.write(json.dumps(r) + "\n")
    cnt = defaultdict(int)
    for r in sel: cnt[r["label"]] += 1
    tot = len(sel); hours = sum(r["duration"] for r in sel) / 3600
    print(f"selected={tot} sources={len(keep_src)} hours={hours:.0f} eval={sum(r['split']=='eval' for r in sel)}")
    for c, k in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {c:18s} {k:7d} {100*k/tot:5.1f}%  (of {n[c]})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
