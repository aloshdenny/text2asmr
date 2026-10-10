#!/usr/bin/env python3
"""Trigger rows for CLAP training from pools whose clips carry a class name (a YouTube title / chapter, a Freesound
search term): a clip is "tier 3" when the fused judges agree with that name (fused P of the class >= --agree), the
same two-signals-agree rule rank_for_people.py applied with Gemini alone. Output rows are train_clap_v8.py
--trigger-manifest rows: {uid, path, info: {tier: 3, title_class, src}}.

--train-exclude also writes the Freesound clips that must never be trained on:
  * licences that forbid it for us: CC-BY-NC (non-commercial) and Sampling+
  * every Freesound sound that is in FSD50K's eval split: FSD50K is built from Freesound, and that split is the external
    human-labelled test set (build_external_eval_set.py), so training on it would contaminate the test

  python build_trigger_manifest.py --fused D:/t2a/fused/merged_v78/fused.jsonl --pool D:/t2a/pool_freesound \\
      --out D:/t2a/pool_freesound/trigger_manifest.jsonl --train-exclude D:/t2a/pool_freesound/train_exclude.txt
"""
from __future__ import annotations
import argparse, csv, glob, json, re
from collections import Counter
from pathlib import Path

TITLE_MAP = {"typing": "tapping", "page turning": "paper rustling", "mouth sounds": "oral sounds"}
NO_TRAIN_LICENCES = ("by-nc", "sampling+")


def safe(uid: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--pool", type=Path, nargs="+", required=True)
    ap.add_argument("--agree", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-exclude", type=Path, default=None)
    a = ap.parse_args()
    fused = {}
    for l in open(a.fused, encoding="utf-8"):
        r = json.loads(l); fused[r["uid"]] = r["probs"]
    fsd_eval = set()
    if a.train_exclude:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("Fhrozen/FSD50k", "labels/eval.csv", repo_type="dataset")
        fsd_eval = {r["fname"].strip() for r in csv.DictReader(open(p, encoding="utf-8"))}
    rows, excl, why = [], [], Counter()
    for d in a.pool:
        files = {Path(f).stem: f for f in glob.glob(str(d / "pool_*" / "clips" / "*.mp3"))}
        for l in open(d / "pool.jsonl", encoding="utf-8"):
            r = json.loads(l); u = r["uid"]
            if u.startswith("fs:"):
                lic = (r.get("license") or "").lower()
                if any(k in lic for k in NO_TRAIN_LICENCES): excl.append(u); why["licence"] += 1; continue
                if u[3:] in fsd_eval: excl.append(u); why["in FSD50K eval (external test)"] += 1; continue
            f, P = files.get(safe(u)), fused.get(u)
            cls = TITLE_MAP.get(r.get("label"), r.get("label"))
            if not f or P is None or cls not in P: why["no audio / not fused / off-menu"] += 1; continue
            if P[cls] < a.agree: why["judges disagree with the name"] += 1; continue
            rows.append({"uid": u, "path": f, "info": {"tier": 3, "title_class": r.get("label"), "src": r.get("src")}})
            why[f"tier 3: {cls}"] += 1
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if a.train_exclude: a.train_exclude.write_text("".join(u + "\n" for u in excl), encoding="utf-8")
    print(f"{len(rows)} trigger rows -> {a.out}; {len(excl)} excluded from training; " +
          ", ".join(f"{k} {v}" for k, v in sorted(why.items())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
