#!/usr/bin/env python3
"""Fill the physical-trigger classes the ASMR corpora barely contain, from labeled public audio.

The soundgasm corpora are vocal ASMR: across 3.15M labeled clips there are only ~3.8k tapping, ~6.5k
crinkling, ~370 scratching and ~46 brushing.  No amount of soundgasm acquisition fixes that, so these
classes are seeded from FSD50K (CC-licensed, per-clip download, AudioSet ontology labels).

This audio is **out of domain**: FSD50K clips are ordinary field/foley recordings, not close-mic binaural
ASMR.  They are written to their own repo with `source: "fsd50k"` on every row so training can cap or
down-weight them, and so an in-domain evaluation never silently becomes an out-of-domain one.

  python3 acquire_triggers.py --work /root/t2a/triggers --per-class 3000

Network-only; one clip on disk at a time, committed to the Hub in batches.
"""
from __future__ import annotations
import argparse, csv, json, os, time
from collections import Counter, defaultdict
from pathlib import Path

SRC = "Fhrozen/FSD50k"
DST = "aoxo/t2a-triggers"

# our ontology class -> FSD50K label names that are actually that sound
CLASS_MAP = {
    "tapping":      ["Tap", "Knock"],
    "scratching":   ["Scratching_(performance_technique)"],
    "crinkling":    ["Crumpling_and_crinkling"],
    "liquid":       ["Liquid", "Pour", "Trickle_and_dribble", "Drip", "Splash_and_splatter", "Fill_(with_liquid)"],
    "page turning": ["Writing"],
    "other sound":  ["Zipper_(clothing)", "Squeak", "Typing"],
}
CAPTIONS = {
    "tapping": ["tapping sounds", "fingers tapping on a surface", "light taps close to the microphone"],
    "scratching": ["scratching sounds", "fingernails scratching a surface", "dry scratching close to the mic"],
    "crinkling": ["crinkling sounds", "crumpling paper or plastic", "crinkly packaging close to the mic"],
    "liquid": ["liquid sounds", "water pouring and trickling", "wet liquid sounds close to the mic"],
    "page turning": ["pages turning", "paper being handled", "writing and paper sounds"],
    "other sound": ["a miscellaneous close-mic sound", "an object handled near the microphone"],
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/root/t2a/triggers"))
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32, help="clips per Hub commit")
    ap.add_argument("--only", default="", help="comma-separated subset of classes")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    api = HfApi()

    try: api.repo_info(DST, repo_type="dataset")
    except Exception:
        api.create_repo(DST, repo_type="dataset", private=False, exist_ok=True); log(f"created {DST}")

    have = {f for f in api.list_repo_files(DST, repo_type="dataset")}
    label_of = {}
    for split in ("dev", "eval"):
        p = hf_hub_download(SRC, f"labels/{split}.csv", repo_type="dataset", cache_dir=str(a.work / "cache"))
        for r in csv.DictReader(open(p)):
            label_of[(split, r["fname"])] = {l.strip() for l in r["labels"].split(",")}

    wanted = {c.strip() for c in a.only.split(",") if c.strip()} or set(CLASS_MAP)
    picks: dict[str, list] = defaultdict(list)
    for (split, fname), labs in label_of.items():
        for cls, names in CLASS_MAP.items():
            if cls in wanted and labs & set(names):
                picks[cls].append((split, fname, sorted(labs)))
                break                                   # one class per clip: no double counting
    log("available: " + ", ".join(f"{c}={len(v)}" for c, v in sorted(picks.items())))

    manifest = a.work / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.open():
            try: done.add(json.loads(line)["path"])
            except Exception: pass

    ops, paths, stats = [], [], Counter()
    with manifest.open("a") as man:
        for cls, items in sorted(picks.items()):
            for split, fname, labs in items[: a.per_class]:
                path_in_repo = f"{cls.replace(' ', '_')}/{fname}.wav"
                if path_in_repo in have or path_in_repo in done: stats[f"{cls}:skip"] += 1; continue
                try:
                    local = hf_hub_download(SRC, f"clips/{split}/{fname}.wav", repo_type="dataset",
                                            cache_dir=str(a.work / "cache"))
                except Exception as e:
                    log(f"  fetch failed {fname}: {type(e).__name__}"); stats[f"{cls}:fail"] += 1; continue
                ops.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=local))
                paths.append(os.path.realpath(local))
                man.write(json.dumps({"path": path_in_repo, "label": cls, "text": CAPTIONS[cls],
                                      "source": "fsd50k", "fsd50k_labels": labs, "domain": "out"}) + "\n")
                stats[cls] += 1
                if len(ops) >= a.batch:
                    for attempt in range(5):
                        try:
                            api.create_commit(DST, repo_type="dataset", operations=ops,
                                              commit_message=f"triggers: +{len(ops)} FSD50K clips"); break
                        except Exception as e:
                            log(f"  commit retry {attempt}: {type(e).__name__} {str(e)[:100]}"); time.sleep(30 * (attempt + 1))
                    for p in paths:
                        try: os.remove(p)
                        except OSError: pass
                    ops, paths = [], []
                    man.flush(); log(f"  {dict(stats)}")
    if ops:
        api.create_commit(DST, repo_type="dataset", operations=ops, commit_message=f"triggers: +{len(ops)} FSD50K clips")
        for p in paths:
            try: os.remove(p)
            except OSError: pass
    api.upload_file(path_or_fileobj=str(manifest), path_in_repo="manifest.jsonl", repo_id=DST, repo_type="dataset",
                    commit_message="trigger manifest")
    log(f"TRIGGERS_DONE {dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
