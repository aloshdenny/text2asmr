#!/usr/bin/env python3
"""Build the CLAP v6 training subset from the per-corpus Qwen3-Omni ledgers.

Balancing is done over (label x creator), not label alone: a class is filled by taking a little from many
creators rather than a lot from few, so oversampling a rare class adds acoustic diversity instead of
repetition.  Whispering / normal speech / silence are background negatives, never targets.

  python3 scripts/build_clap_v6_manifest.py --out label_tool/v6 --per-class 30000 --bg 60000

Outputs <out>/subset_<corpus>.jsonl (prep_clap_v2 input: uid,label,text,split,source,start,duration)
and <out>/stats.json.  Geometry comes from t2a-mommy's gate index where available, and is recomputed
from the alignment JSONs for the remaining sources.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import hf_hub_download
from text2asmr.data.segment import load_alignment, split_alignment

MOMMY, DADDY = "aoxo/t2a-mommy", "aoxo/t2a-daddy"
TARGETS = ["kissing", "mouth sounds", "breathing", "moaning"]
BG_LABELS = ["whispering", "normal speech", "silence"]
BG = "__bg__"
CAPTIONS = {
    "kissing": ["kissing sounds close to the microphone", "soft wet kisses", "someone kissing the mic"],
    "mouth sounds": ["wet mouth sounds", "licking and lip smacking", "tongue and mouth noises close to the mic"],
    "breathing": ["close breathing into the microphone", "soft breaths and panting", "audible breathing, no words"],
    "moaning": ["a person moaning softly", "quiet moans and whimpers", "moaning voice close to the mic"],
    BG: ["a person talking, normal speech", "someone whispering words softly", "quiet room tone, silence"],
}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ledger_files() -> list[tuple[str, str]]:
    """Every label ledger on the Hub, including the per-shard files that parallel labeling runs write
    (labels/qwen3omni_expansion.s0.jsonl etc.) -- missing those silently trains on a subset."""
    from huggingface_hub import HfApi
    api = HfApi(); out = []
    for repo in (MOMMY, DADDY):
        out += [(repo, f) for f in api.list_repo_files(repo, repo_type="dataset")
                if f.startswith("labels/qwen3omni") and f.endswith(".jsonl")]
    return out


def load_ledgers(cache: str) -> list[dict]:
    """Every label row we have, as {uid, label, source, creator, repo}."""
    rows = []
    for repo, name in ledger_files():
        try: p = hf_hub_download(repo, name, repo_type="dataset", cache_dir=cache)
        except Exception as e: log(f"missing {repo}:{name} ({type(e).__name__})"); continue
        n = 0
        for line in open(p):
            r = json.loads(line); uid = r["uid"]
            src = r.get("source")
            if not src:
                if ".m4a_" not in uid: continue
                src = uid.rsplit(".m4a_", 1)[0] + ".m4a"
            rows.append({"uid": uid, "label": r["label"], "source": src,
                         "creator": src.split("/")[0], "repo": repo}); n += 1
        log(f"{repo}:{name} -> {n} rows")
    return rows


def pick(rows: list[dict], quota: int, per_creator_frac: float, per_source: int, rng: random.Random) -> list[dict]:
    """Round-robin over creators (and over sources within a creator) until the quota is met.

    A creator can never contribute more than per_creator_frac of the class, so a class is never
    one creator's microphone."""
    by_creator: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows: by_creator[r["creator"]][r["source"]].append(r)
    for c in by_creator:
        for s in by_creator[c]: rng.shuffle(by_creator[c][s])
    creators = sorted(by_creator); rng.shuffle(creators)
    cap = max(1, int(quota * per_creator_frac))
    took: dict[str, int] = defaultdict(int); out, exhausted = [], set()
    while len(out) < quota and len(exhausted) < len(creators):
        for c in creators:
            if c in exhausted or len(out) >= quota: continue
            if took[c] >= cap: exhausted.add(c); continue
            srcs = [s for s, v in by_creator[c].items() if v]
            if not srcs: exhausted.add(c); continue
            s = srcs[rng.randrange(len(srcs))]
            n = min(per_source, len(by_creator[c][s]), cap - took[c], quota - len(out))
            out.extend(by_creator[c][s][:n]); by_creator[c][s] = by_creator[c][s][n:]
            took[c] += n
    return out


def geometry_from_index(cache: str, wanted: set[str]) -> dict[str, tuple[float, float]]:
    """uid -> (cut_start, cut_duration) from t2a-mommy's gate index."""
    geo = {}
    try: p = hf_hub_download(MOMMY, "labels/clap_gate_index.jsonl", repo_type="dataset", cache_dir=cache)
    except Exception as e: log(f"no gate index ({type(e).__name__})"); return geo
    for line in open(p):
        r = json.loads(line)
        if r["uid"] in wanted:
            start = r.get("cut_start", max(0.0, r["start"] - 1.0))
            dur = r.get("cut_duration", min(8.0, r["duration"] + 2.0))
            geo[r["uid"]] = (float(start), float(dur))
    log(f"gate index covered {len(geo)}/{len(wanted)} uids")
    return geo


def geometry_from_alignments(repo: str, sources: list[str], cache: str, workers: int) -> dict[str, tuple[float, float]]:
    """Recompute gap geometry for sources the index does not cover (same cuts the labeler saw)."""
    geo: dict[str, tuple[float, float]] = {}
    def one(src):
        try: entries = load_alignment(hf_hub_download(repo, src + ".json", repo_type="dataset", cache_dir=cache))
        except Exception: return []
        return [(sp.uid, max(0.0, sp.start - 1.0), min(8.0, sp.duration + 2.0))
                for sp in split_alignment(entries, src) if sp.kind == "trigger_candidate"]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, got in enumerate(ex.map(one, sources)):
            for uid, s, d in got: geo[uid] = (s, d)
            if i % 250 == 0: log(f"  alignments {i}/{len(sources)} -> {len(geo)} uids")
    return geo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=30000, help="clips per target class (both corpora combined)")
    ap.add_argument("--bg", type=int, default=60000, help="background clips in total")
    ap.add_argument("--per-creator-frac", type=float, default=0.03)
    ap.add_argument("--per-source", type=int, default=60, help="max clips taken from one source file in one pass")
    ap.add_argument("--eval-creator-frac", type=float, default=0.08)
    ap.add_argument("--cache", default=os.environ.get("T2A_CACHE", "hfcache"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-creators-file", type=Path, default=None,
                    help="pin the held-out creators (one 'repo\tcreator' per line) so metrics stay comparable "
                         "across versions; written on first use, read afterwards")
    ap.add_argument("--already", type=Path, default=None,
                    help="an existing mels index.jsonl; its uids are written to <out>/delta_*.jsonl so prep only "
                         "has to cut what is new")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    rows = load_ledgers(a.cache)
    log(f"{len(rows)} label rows, {len({r['creator'] for r in rows})} creators")
    by_label = defaultdict(list)
    for r in rows: by_label[r["label"]].append(r)
    log("available: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_label.items(), key=lambda kv: -len(kv[1]))))

    # per-corpus quotas so the male side is never drowned out by the (larger) female side
    picked: list[dict] = []
    for label in TARGETS:
        for repo in (MOMMY, DADDY):
            pool = [r for r in by_label.get(label, []) if r["repo"] == repo]
            want = a.per_class // 2
            got = pick(pool, want, a.per_creator_frac, a.per_source, rng)
            for r in got: r["target"] = label
            picked.extend(got)
            log(f"{label:14} {repo.split('/')[-1]:10} wanted {want:6} got {len(got):6} "
                f"from {len({r['creator'] for r in got})} creators")
    for label in BG_LABELS:
        for repo in (MOMMY, DADDY):
            pool = [r for r in by_label.get(label, []) if r["repo"] == repo]
            want = a.bg // (2 * len(BG_LABELS))
            got = pick(pool, want, a.per_creator_frac, a.per_source, rng)
            for r in got: r["target"] = BG
            picked.extend(got)
            log(f"{label:14} {repo.split('/')[-1]:10} (bg) wanted {want:6} got {len(got):6}")

    # creator-split: a creator is either train or eval, never both, per corpus
    creators = sorted({(r["repo"], r["creator"]) for r in picked})
    if a.eval_creators_file and a.eval_creators_file.exists():
        eval_creators = {tuple(l.rstrip("\n").split("\t")) for l in a.eval_creators_file.open() if l.strip()}
        log(f"{len(creators)} creators in subset; {len(eval_creators)} eval creators pinned from {a.eval_creators_file}")
    else:
        rng.shuffle(creators)
        n_eval = int(len(creators) * a.eval_creator_frac)
        eval_creators = set(creators[:n_eval])
        log(f"{len(creators)} creators in subset; {len(eval_creators)} held out for eval")
        if a.eval_creators_file:
            a.eval_creators_file.write_text("".join(f"{r}\t{c}\n" for r, c in sorted(eval_creators)))
            log(f"wrote {a.eval_creators_file}")

    # geometry
    wanted_mommy = {r["uid"] for r in picked if r["repo"] == MOMMY}
    geo = geometry_from_index(a.cache, wanted_mommy)
    missing = defaultdict(set)
    for r in picked:
        if r["uid"] not in geo: missing[r["repo"]].add(r["source"])
    for repo, srcs in missing.items():
        log(f"{repo}: recomputing geometry for {len(srcs)} sources")
        geo.update(geometry_from_alignments(repo, sorted(srcs), a.cache, a.workers))

    stats = defaultdict(lambda: defaultdict(int)); written = 0
    handles = {MOMMY: open(a.out / "subset_mommy.jsonl", "w"), DADDY: open(a.out / "subset_daddy.jsonl", "w")}
    for r in picked:
        g = geo.get(r["uid"])
        if not g: stats["dropped"]["no_geometry"] += 1; continue
        split = "eval" if (r["repo"], r["creator"]) in eval_creators else "train"
        handles[r["repo"]].write(json.dumps({
            "uid": r["uid"], "label": r["target"], "text": CAPTIONS[r["target"]], "split": split,
            "source": r["source"], "start": round(g[0], 3), "duration": round(g[1], 3),
            "creator": r["creator"], "raw_label": r["label"], "repo": r["repo"]}) + "\n")
        stats[split][r["target"]] += 1; written += 1
    for h in handles.values(): h.close()

    if a.already and a.already.exists():
        have = set()
        for line in a.already.open():
            try: have.add(json.loads(line)["uid"])
            except Exception: pass
        for corpus in ("mommy", "daddy"):
            src = a.out / f"subset_{corpus}.jsonl"; dst = a.out / f"delta_{corpus}.jsonl"
            n = 0
            with src.open() as fi, dst.open("w") as fo:
                for line in fi:
                    if json.loads(line)["uid"] not in have: fo.write(line); n += 1
            log(f"delta_{corpus}: {n} clips not yet in {a.already}")
    json.dump({k: dict(v) for k, v in stats.items()}, open(a.out / "stats.json", "w"), indent=2)
    log(f"wrote {written} rows -> {a.out}")
    log("train: " + json.dumps(dict(stats["train"])) + "  eval: " + json.dumps(dict(stats["eval"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
