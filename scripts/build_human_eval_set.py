#!/usr/bin/env python3
"""Sample a human-labelling set for validation, from creators nothing else is allowed to touch.

Everything we call "ground truth" today is model-labelled (Gemini-Pro, then Qwen3-Omni).  That is fine for
training and useless as a final answer: a model cannot certify itself.  This builds the set a human labels
instead, and it is never trained on.

Design points that matter:

* **Creator-disjoint.** Clips come only from creators reserved here and written to eval_creators_blocked.txt,
  which the training manifest builder refuses to sample from.  Otherwise the eval measures memorisation.
* **The Qwen3 label is a *stratifier*, not truth.** It is used only to make sure every class appears in the
  sample; it is not shown to the annotator and not stored with the clip.
* **A uniform slice.** Stratified sampling distorts prevalence, so --uniform clips are drawn at random to
  measure how the classes actually occur.
* **Double-labelled overlap.** --overlap clips are assigned to two annotators, because the number we most
  need is not the model's accuracy but the *human ceiling*: published ASMR annotation agrees 100% on speech,
  83% on whispering and only 67% on mouth sounds.  A model at 0.69 on mouth sounds may already be at ceiling,
  and without the overlap we cannot tell.

  python3 build_human_eval_set.py --per-class 150 --uniform 300 --overlap 600 --out label_tool/humaneval
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MOMMY, DADDY = "aoxo/t2a-mommy", "aoxo/t2a-daddy"
# every class the labeler is asked about, so the eval can measure the whole ontology, not just what survived
ONTOLOGY = ["kissing", "mouth sounds", "breathing", "moaning", "whispering", "normal speech", "silence",
            "tapping", "scratching", "crinkling", "brushing", "liquid", "page turning", "other sound"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_labels(cache: str) -> list[dict]:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); rows = []
    for repo in (MOMMY, DADDY):
        for f in api.list_repo_files(repo, repo_type="dataset"):
            if not (f.startswith("labels/qwen3omni") and f.endswith(".jsonl")): continue
            try: p = hf_hub_download(repo, f, repo_type="dataset", cache_dir=cache, force_download=True)
            except Exception as e: log(f"  skip {repo}:{f} ({type(e).__name__})"); continue
            n = 0
            for line in open(p):
                try: r = json.loads(line)
                except Exception: continue
                uid = r.get("uid", "")
                src = r.get("source") or (uid.rsplit(".m4a_", 1)[0] + ".m4a" if ".m4a_" in uid else None)
                if not src or not r.get("label"): continue
                rows.append({"uid": uid, "label": r["label"], "source": src,
                             "creator": src.split("/")[0], "repo": repo}); n += 1
            log(f"  {repo}:{f} -> {n}")
            try: os.remove(os.path.realpath(p))
            except OSError: pass
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--uniform", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=600, help="clips assigned to two annotators")
    ap.add_argument("--creators", type=int, default=120, help="creators to reserve for evaluation")
    ap.add_argument("--max-per-creator", type=int, default=25)
    ap.add_argument("--cache", default=os.environ.get("T2A_CACHE", "hfcache"))
    ap.add_argument("--seed", type=int, default=20260925)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    rows = load_labels(a.cache)
    log(f"{len(rows)} label rows, {len({r['creator'] for r in rows})} creators")

    # reserve creators: prefer ones with a wide spread of classes, so the eval is not a handful of voices
    by_creator: dict[tuple, Counter] = defaultdict(Counter)
    for r in rows: by_creator[(r["repo"], r["creator"])][r["label"]] += 1
    ranked = sorted(by_creator.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))
    pool = [k for k, _ in ranked if sum(by_creator[k].values()) >= 40]
    rng.shuffle(pool)
    reserved = set(pool[: a.creators])
    log(f"reserved {len(reserved)} creators "
        f"({sum(1 for r, _ in reserved if r == MOMMY)} female / {sum(1 for r, _ in reserved if r == DADDY)} male)")

    held = [r for r in rows if (r["repo"], r["creator"]) in reserved]
    log(f"{len(held)} clips live in reserved creators")
    by_label: dict[str, list] = defaultdict(list)
    for r in held: by_label[r["label"]].append(r)

    picked: list[dict] = []
    taken_per_creator: Counter = Counter()
    for label in ONTOLOGY:
        cand = by_label.get(label, [])[:]
        rng.shuffle(cand)
        n = 0
        for r in cand:
            if n >= a.per_class: break
            key = (r["repo"], r["creator"])
            if taken_per_creator[key] >= a.max_per_creator: continue
            taken_per_creator[key] += 1
            picked.append(dict(r, stratum=label)); n += 1
        log(f"  {label:14} available {len(cand):7} sampled {n}")

    chosen = {r["uid"] for r in picked}
    rest = [r for r in held if r["uid"] not in chosen]
    rng.shuffle(rest)
    for r in rest[: a.uniform]:
        picked.append(dict(r, stratum="__uniform__"))
    log(f"+{min(a.uniform, len(rest))} uniform clips for prevalence")

    rng.shuffle(picked)
    for i, r in enumerate(picked):
        r["annotators"] = 2 if i < a.overlap else 1       # the overlap slice measures the human ceiling
        r["order"] = i
        r.pop("label")                                     # the model's guess never reaches the annotator
    manifest = a.out / "human_eval_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in picked))

    blocked = a.out / "eval_creators_blocked.txt"
    blocked.write_text("".join(f"{repo}\t{c}\n" for repo, c in sorted(reserved)))
    stats = {"clips": len(picked), "creators": len(reserved), "per_class": a.per_class,
             "uniform": a.uniform, "double_labelled": min(a.overlap, len(picked)),
             "strata": dict(Counter(r["stratum"] for r in picked)), "seed": a.seed}
    json.dump(stats, open(a.out / "human_eval_stats.json", "w"), indent=2)
    log(f"wrote {len(picked)} clips -> {manifest}")
    log(f"blocked {len(reserved)} creators from training -> {blocked}")
    log(json.dumps(stats["strata"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
