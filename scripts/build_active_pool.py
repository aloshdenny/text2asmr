#!/usr/bin/env python3
"""Pick the next batch for the LLM labellers (label_pool.py --pool-file): active learning over every mel row.

probe_fused.py scored all ~279k CLAP rows with a probe trained on the fused labels. The rows worth an LLM's time are
the ones the probe is least sure of (some class near P 0.5), or that probably hold a rare sound the fusion has few
examples of. Rows already in any pool are skipped, at most --per-rec windows come from one recording, and each row is
written with the repo / recording / start label_pool.py needs to cut it from the Hub.

  python build_active_pool.py --rows D:/t2a/probe/rows.jsonl --pools D:/t2a/pool D:/t2a/pool_yt D:/t2a/pool_spray \\
      D:/t2a/pool_mined D:/t2a/anchors/user D:/t2a/anchors/adi --n 8000 --out D:/t2a/pool_active/pool.jsonl
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path

RARE = ["microphone touching", "sticky", "fabric rustling", "paper rustling", "cutting", "brushing", "liquid", "spraying", "scratching"]
CORPORA = ("aoxo/t2a-mommy", "aoxo/t2a-daddy")
YT_REPO = "aoxo/asmr-yt-chapters"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True, help="probe_fused.py rows.jsonl")
    ap.add_argument("--pools", type=Path, nargs="*", default=[], help="pool dirs whose pool.jsonl rows are already labelled")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--per-rec", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    seen = set()
    for d in a.pools:
        f = d / "pool.jsonl"
        if f.exists(): seen |= {json.loads(l)["uid"] for l in open(f, encoding="utf-8")}
    from huggingface_hub import HfApi
    owner = {}
    for repo in CORPORA:                                       # recording path -> the corpus repo that holds it
        for f in HfApi().list_repo_files(repo, repo_type="dataset"):
            if f.endswith(".m4a"): owner.setdefault(f, repo)
    cands = []
    for l in open(a.rows, encoding="utf-8"):
        r = json.loads(l)
        u = r["uid"]
        if u in seen: continue
        p = r.get("p", {})
        unsure = max((1 - abs(2 * v - 1) for v in p.values()), default=0.0)
        rare = max((p.get(c, 0.0) for c in RARE), default=0.0)
        if u.startswith("yt:"):
            _, vid, start = u.split(":")
            row = {"repo": YT_REPO, "rec": vid, "start": float(start), "src": "yt_chapter"}
        else:
            rec = re.sub(r"_\d+$", "", u)
            if rec not in owner: continue
            row = {"repo": owner[rec], "rec": rec, "start": int(u.rsplit("_", 1)[1]) / 1000 - 0.5, "src": "indomain"}
        cands.append((max(unsure, rare), {"uid": u, "label": r.get("label") or "__bg__", **row, "probe_unsure": round(unsure, 3), "probe_rare": round(rare, 3)}))
    cands.sort(key=lambda x: -x[0])
    per_rec, picks = Counter(), []
    for score, row in cands:
        if per_rec[row["rec"]] >= a.per_rec: continue
        per_rec[row["rec"]] += 1
        picks.append(row)
        if len(picks) >= a.n: break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in picks), encoding="utf-8")
    print(f"{len(cands)} unlabelled candidates; picked {len(picks)} from {len(per_rec)} recordings "
          f"({Counter(r['src'] for r in picks)}); score range {cands[0][0]:.2f} .. {picks[-1] and max(picks[-1]['probe_unsure'], picks[-1]['probe_rare']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
