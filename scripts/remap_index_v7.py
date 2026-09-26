#!/usr/bin/env python3
"""Rewrite a prepped mel index for the v7 ontology without re-cutting a single clip.

Mel shards are label-agnostic: the audio features sit in shard_*.f16 and every label lives in index.jsonl.
So the kissing+mouth-sounds merge decided on 2026-09-26 costs one pass over a text file rather than a
re-prep of 186k windows (23 GB, several GPU-hours, and disk we do not have).

The original label is kept as `raw_label` on every row, so the merge is reversible if a human-labelled set
later shows the split is real.

  python3 remap_index_v7.py --index ~/t2a/v6/mels/index.jsonl --out ~/t2a/v7/index.jsonl
"""
from __future__ import annotations
import argparse, json, shutil, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text2asmr.io_guard import preflight

# v7 ontology: five independent judges could not separate kissing from mouth sounds (mean 66.3% on three
# options, chance 33%); merged they agree 82.8%. See docs/BEAT_DEEPASMR.md section 8.
MERGE = {"kissing": "oral sounds", "mouth sounds": "oral sounds"}
CAPTIONS = {
    "oral sounds": ["wet mouth and kissing sounds close to the microphone", "lips, tongue and saliva sounds",
                    "soft kisses, licking and lip smacking near the mic"],
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True, help="existing prepped index.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="v7 index (written next to the same shards)")
    ap.add_argument("--link-shards", action="store_true",
                    help="symlink the shard files beside the new index so a trainer can point at one directory")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    preflight(a.out, note="v7 index")

    before, after = Counter(), Counter()
    n = 0
    with a.index.open() as fi, a.out.open("w") as fo:
        for line in fi:
            try: r = json.loads(line)
            except Exception: continue
            lab = r.get("label")
            before[lab] += 1
            new = MERGE.get(lab, lab)
            if new != lab:
                r["raw_label"] = lab          # reversible: the finer distinction is never discarded
                r["label"] = new
                if new in CAPTIONS: r["text"] = CAPTIONS[new]
            after[new] += 1
            fo.write(json.dumps(r) + "\n"); n += 1
    log(f"{n} rows -> {a.out}")
    log("before: " + ", ".join(f"{k}={v}" for k, v in before.most_common()))
    log("after : " + ", ".join(f"{k}={v}" for k, v in after.most_common()))

    if a.link_shards:
        linked = 0
        for sh in sorted(a.index.parent.glob("shard_*.f16")):
            dst = a.out.parent / sh.name
            if not dst.exists():
                dst.symlink_to(sh)            # symlink, not copy: 23 GB stays where it is
                linked += 1
        log(f"linked {linked} shard files beside {a.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
