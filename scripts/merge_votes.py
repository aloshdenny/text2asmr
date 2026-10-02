#!/usr/bin/env python3
"""Large-scale weak supervision, step 1b: merge every pool's labeller files into one file per labeller for fuse_labels.py.

Pools: label_pool.py out dirs (pool.jsonl + gemini/mimo/voxtral jsonl) with pool_local_labels.py's ast/energy/clap.
A pool.jsonl row is the pipeline's vote for v7s2 clips and the chapter title's vote for YouTube windows ("yt:" uids).
Chapter votes come from the window's chapter title (--windows), so a title that names a spray ("Spray bottle",
"water spraying") votes "spraying" even where the pool filed the window under liquid. Every chapter vote was offered
the full menu; the other labellers keep the menu their file recorded (none = the 19 labels before spraying).

  python merge_votes.py --pools pool yt spray anch_user anch_adi --windows windows.jsonl singles.jsonl --out merged/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from fuse_labels import CHAPTER, LABELS, V7

NAMES = {"gemini-3.1-pro": "gemini", "mimo-v2.6-flash": "mimo", "voxtral": "voxtral", "ast": "ast", "energy": "energy", "clap": "clap"}


def chapter_vote(label: str | None, title: str) -> list[str]:
    if "spray" in title.lower(): return ["spraying"]
    label = {"mouth sounds": "oral sounds"}.get(label, label)
    return [label] if label in CHAPTER else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", type=Path, nargs="+", required=True)
    ap.add_argument("--windows", type=Path, nargs="*", default=[], help="YT window lists with chapter titles")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    titles = {}
    for f in a.windows:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); titles.setdefault(r["uid"], r.get("chapter") or "")
    out: dict = {}
    for d in a.pools:
        for f in sorted(d.glob("*.jsonl")):
            stem = f.stem
            for l in open(f, encoding="utf-8"):
                r = json.loads(l); uid = r.get("clip") or r.get("uid")
                if stem == "pool":
                    if r.get("label") in (None, "uniform"): continue
                    if uid.startswith("yt:"):
                        out.setdefault("chapter", {})[uid] = {"labels": chapter_vote(r["label"], r.get("chapter") or titles.get(uid, "")),
                                                             "menu": LABELS}
                    else:
                        out.setdefault("pipeline", {})[uid] = {"labels": [r["label"]] if r["label"] in V7 else []}
                elif stem in NAMES:
                    if str(r.get("raw", "")).startswith("ERROR"): continue
                    out.setdefault(NAMES[stem], {})[uid] = {"labels": r["labels"], **({"menu": r["menu"]} if r.get("menu") else {})}
    a.out.mkdir(parents=True, exist_ok=True)
    for name, d in out.items():
        (a.out / f"{name}.jsonl").write_text("".join(json.dumps({"uid": u, **v}) + "\n" for u, v in d.items()), encoding="utf-8")
        sp = sum("spraying" in v["labels"] for v in d.values()); menu = sum("menu" in v for v in d.values())
        print(f"{name:9} {len(d):6} clips  spraying {sp:4}  with 20-label menu {menu}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
