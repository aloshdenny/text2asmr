#!/usr/bin/env python3
"""CLAP v7 stage 2 data: in-domain vocal mels + YouTube physical-tail mels, one directory, weighted rows.

Weak supervision at scale only averages noise out while the true class is the plurality inside its label.
So every row gets w = the measured precision of its (source, class) cell, and cells below --min-precision
are dropped as a *source*, never clip by clip:

  in-domain Qwen3-Omni labels ..... w = 1.0 (stage-1 baseline; 5-judge agreement 83-100% on these classes)
  YouTube chapters, Gemini-judged . keep-rate of the cell, read live from the Gemini filter's output
      a window Gemini kept ........ w = GEMINI_ACC[class] (its accuracy on human-labelled clips)
      a window Gemini dropped ..... removed
      not judged yet .............. w = cell keep-rate, or removed if that is below --min-precision
  YouTube chapters, unmeasured .... w = --prior (Gemini cannot hear these classes; see class_gate.py)

Shards are symlinked, not copied: the YouTube shards get a shard-number offset so both sets live in one dir.

  python3 build_v7_stage2.py --indomain ~/t2a/v7/mels --yt ~/t2a/v7s2/ytmels --out ~/t2a/v7s2/data
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text2asmr.io_guard import preflight

BG = "__bg__"
MERGE = {"kissing": "oral sounds", "mouth sounds": "oral sounds", "licking": "oral sounds"}   # v7 ontology
GEMINI_ACC = {"brushing": 0.88, "crinkling": 0.80, "liquid": 0.75, "tapping": 0.57}          # class_gate cutext
ORAL_TEXT = ["wet mouth and kissing sounds close to the microphone", "lips, tongue and saliva sounds",
             "soft kisses, licking and lip smacking near the mic"]
SHARD_OFFSET = 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indomain", type=Path, required=True)
    ap.add_argument("--yt", type=Path, required=True, help="dir holding the YouTube index.jsonl + shard_*.f16")
    ap.add_argument("--gemini", type=Path, default=None, help="gemini_judged.jsonl from gemini_clean_yt.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-precision", type=float, default=0.4, help="drop a (source, class) cell below this")
    ap.add_argument("--prior", type=float, default=0.5, help="weight for YouTube cells nobody can measure yet")
    ap.add_argument("--min-train", type=int, default=800, help="drop a class with fewer effective train rows")
    ap.add_argument("--min-eval", type=int, default=40)
    a = ap.parse_args()
    yt_idx = next(a.yt.rglob("index.jsonl"))
    yt_dir = yt_idx.parent
    a.out.mkdir(parents=True, exist_ok=True)
    preflight(a.out / "index.jsonl", note="v7 stage 2 index")

    # ---- Gemini verdicts -> per-window keep/drop, per-class keep rate ----
    verdict, rate = {}, {}
    if a.gemini and a.gemini.exists():
        tally = defaultdict(Counter)
        for l in a.gemini.open():
            try: r = json.loads(l)
            except Exception: continue
            if r.get("pred") is None: continue
            verdict[r["uid"]] = bool(r.get("keep"))
            tally[r["label"]][bool(r.get("keep"))] += 1
        rate = {c: t[True] / max(1, t[True] + t[False]) for c, t in tally.items() if sum(t.values()) >= 100}
        print("gemini keep-rate per chapter class: " + ", ".join(f"{c} {v:.2f} (n={sum(tally[c].values())})"
                                                                 for c, v in sorted(rate.items())))

    rows, why = [], Counter()
    for l in (a.indomain / "index.jsonl").open():
        r = json.loads(l); r["w"] = 1.0; r["src"] = "indomain"; rows.append(r)
    n_in = len(rows)
    overlap = 0
    for l in yt_idx.open():
        r = json.loads(l)
        lab = MERGE.get(r["label"], r["label"])
        if lab != r["label"]:
            r["raw_label"] = r["label"]; r["label"] = lab
            if lab == "oral sounds": r["text"] = ORAL_TEXT
        r["shard"] = int(r["shard"]) + SHARD_OFFSET; r["src"] = "yt_chapter"
        if lab == BG:
            r["w"] = 1.0
        elif r["uid"] in verdict:
            overlap += 1
            if not verdict[r["uid"]]: why[f"{lab}: gemini dropped"] += 1; continue
            r["w"] = GEMINI_ACC.get(lab, 0.8)
        elif lab in rate:
            if rate[lab] < a.min_precision: why[f"{lab}: cell precision {rate[lab]:.2f} < {a.min_precision}"] += 1; continue
            r["w"] = round(rate[lab], 3)
        else:
            r["w"] = a.prior
        rows.append(r)
    print(f"in-domain rows {n_in}; youtube rows kept {len(rows) - n_in}; gemini verdicts matched {overlap}")
    for k, v in why.most_common(): print(f"  removed {v:6d}  {k}")

    # ---- drop classes too thin to train or to evaluate ----
    eff = defaultdict(float); ev = Counter()
    for r in rows:
        if r["label"] == BG: continue
        if r["split"] == "train": eff[r["label"]] += r["w"]
        else: ev[r["label"]] += 1
    thin = {c for c in set(eff) | set(ev) if eff[c] < a.min_train or ev[c] < a.min_eval}
    if thin: print("dropping thin classes: " + ", ".join(f"{c} (eff train {eff[c]:.0f}, eval {ev[c]})" for c in sorted(thin)))
    rows = [r for r in rows if r["label"] not in thin]

    # ---- one directory: link both shard sets, write the merged index ----
    linked = 0
    for src, off in ((a.indomain, 0), (yt_dir, SHARD_OFFSET)):
        for sh in sorted(src.glob("shard_*.f16")):
            n = int(sh.stem.split("_")[1]) + off
            dst = a.out / f"shard_{n:04d}.f16"
            if not dst.exists(): dst.symlink_to(sh.resolve()); linked += 1
    with (a.out / "index.jsonl").open("w") as fo:
        for r in rows: fo.write(json.dumps(r) + "\n")
    tab = defaultdict(lambda: [0, 0.0, 0])
    for r in rows:
        t = tab[r["label"]]
        if r["split"] == "train": t[0] += 1; t[1] += r["w"]
        else: t[2] += 1
    print(f"\n{'class':22} {'train':>8} {'effective':>10} {'eval':>6}")
    for c, (n, e, v) in sorted(tab.items(), key=lambda kv: -kv[1][1]):
        print(f"{c:22} {n:8d} {e:10.0f} {v:6d}")
    print(f"\nlinked {linked} shards; {len(rows)} rows -> {a.out / 'index.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
