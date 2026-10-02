#!/usr/bin/env python3
"""CLAP v7.1 training index: v7's mel rows, relabelled by the fusion where clips were labelled and the probe elsewhere.

v7 stage 2 trained every row on one weak label (its chapter title or the pipeline's class), weighted by that
source's measured precision. The fusion says several of those cells are mostly something else (paper-rustling
chapter windows: ~4% paper), and a per-cell weight cannot fix that: it only changes how often the class is drawn,
not which of its rows are right. Here each row gets its own probability per class:
  * fused P (fuse_labels.py) for the ~11k clips machines and people labelled, probe P (probe_fused.py) for the rest
  * a row is a positive of every target class with P >= --pos, with w = P (train_clap_v3 keeps a row with prob. w)
  * a row is background when every target class has P < --neg
  * anything in between is left out
Shards and splits are v7's (YouTube shards keep their +1000 offset); prompts are v7's, new classes get the same
template plus the ontology's probes.

  python build_v71_index.py --index v7s2/index.jsonl --yt-index emb/yt.jsonl --probe probe/rows.jsonl --fused fused.jsonl --out v71/
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

BG = "__bg__"
V7 = ["breathing", "crinkling", "cutting", "fabric rustling", "microphone touching", "moaning", "oral sounds",
      "paper rustling", "scratching", "sticky", "tapping"]
NEW = ["brushing", "liquid", "spraying"]
SHARD_OFFSET = 1000
PROBES = {"brushing": ["brushing sounds", "a soft brush moving across a microphone", "bristles sweeping"],
          "liquid": ["pouring liquid", "water sloshing in a bottle", "stirring a drink"],
          "spraying": ["a spray bottle spraying mist", "spritzing water from a spray bottle", "aerosol spray hissing"]}


def template(c: str) -> list[str]:
    return [f"ASMR {c}, close-mic binaural recording, no speech", f"the sound of {c}", f"{c} sounds close to a microphone",
            f"soft {c} ASMR trigger"] + PROBES.get(c, [])


def class_p(p: dict, c: str) -> float:
    return max(p.get("oral sounds", 0.0), p.get("kissing", 0.0)) if c == "oral sounds" else p.get(c, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True, help="v7 stage-2 index.jsonl (rows, shards, prompts)")
    ap.add_argument("--yt-index", type=Path, default=None, help="every YouTube mel row (embed_index_clap.py yt.jsonl), to add windows v7 left out")
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pos", type=float, default=0.5)
    ap.add_argument("--neg", type=float, default=0.15)
    ap.add_argument("--min-effective", type=float, default=300, help="drop a class with fewer effective train rows (sum of w)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, texts = {}, {}
    for l in open(a.index, encoding="utf-8"):
        r = json.loads(l)
        if r["label"] != BG and r.get("text"): texts.setdefault(r["label"], r["text"])
        rows.setdefault(r["uid"], {k: r[k] for k in ("uid", "split", "shard", "row", "src")})
    added = 0
    if a.yt_index:
        for l in open(a.yt_index, encoding="utf-8"):
            r = json.loads(l)
            if r["uid"] in rows: continue
            rows[r["uid"]] = {"uid": r["uid"], "split": r.get("split") or "train", "shard": r["shard"] + SHARD_OFFSET, "row": r["row"], "src": "yt_chapter"}
            added += 1
    F = {json.loads(l)["uid"]: json.loads(l)["probs"] for l in open(a.fused, encoding="utf-8")}
    P = {}
    for l in open(a.probe, encoding="utf-8"):
        r = json.loads(l); P[r["uid"]] = r["p"]
    print(f"{len(rows)} mel rows ({added} YouTube windows v7 left out); fused for {sum(u in F for u in rows)}, probe for {sum(u in P and u not in F for u in rows)}")
    classes = V7 + NEW
    out, eff, n_bg, n_drop, by = [], Counter(), 0, 0, defaultdict(Counter)
    for u, r in rows.items():
        p = F.get(u) or P.get(u)
        if p is None: n_drop += 1; continue
        pos = [(c, class_p(p, c)) for c in classes if class_p(p, c) >= a.pos]
        if pos:
            for c, pc in pos:
                out.append({**r, "label": c, "w": round(pc, 3), "by": "fused" if u in F else "probe"})
                if r["split"] == "train": eff[c] += pc
                by[c]["fused" if u in F else "probe"] += 1
        elif max(class_p(p, c) for c in classes) < a.neg:
            out.append({**r, "label": BG, "w": 1.0, "by": "fused" if u in F else "probe"}); n_bg += 1
        else: n_drop += 1
    keep = {c for c in classes if eff[c] >= a.min_effective}
    for c in classes:
        print(f"{c:20} effective train rows {eff[c]:8.0f}  rows from fusion {by[c]['fused']:5}  from probe {by[c]['probe']:6}  {'' if c in keep else 'DROPPED (too few)'}")
    out = [r for r in out if r["label"] == BG or r["label"] in keep]
    for r in out:
        r["text"] = texts.get(r["label"]) or template(r["label"]) if r["label"] != BG else texts.get(BG, [])
    with open(a.out / "index.jsonl", "w", encoding="utf-8") as fh:
        for r in out: fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} rows ({n_bg} background, {n_drop} ambiguous left out) -> {a.out / 'index.jsonl'}")
    print("splits:", dict(Counter((r["split"], r["label"] == BG) for r in out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
