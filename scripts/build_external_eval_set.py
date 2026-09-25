#!/usr/bin/env python3
"""Assemble a validation set out of audio that humans already labelled, so nothing here is our own guess.

We have no annotators, and every label we produced is model-made (Gemini-Pro, then Qwen3-Omni).  Rather than
validate a model against another model, this pulls clips whose labels came from published human annotation
campaigns and maps their classes onto our ontology:

  FSD50K  -- eval split only (the dev split seeded aoxo/t2a-triggers, so it is training data)
  ESC-50  -- 2,000 clips, 5 human-validated folds, never used here for training
  AudioSet eval -- 3 raters per segment, rated blind to title/metadata

What this can and cannot do, stated plainly:

* It is **out of domain**.  These are field and foley recordings, not close-mic binaural ASMR, so scores here
  measure "does the model hear the sound class as humans define it", not "does it work on our corpus".
  Every row carries domain="out" and its source dataset, and results must be reported per source.
* **kissing has no human-labelled source anywhere** -- AudioSet's 632 classes contain no kiss class, and
  AudioCaps has 3 mentions, none of them ASMR.  moaning and mouth sounds only have semantic proxies whose
  meaning differs from ours (AudioSet "Wail, moan" is distress; "Chewing, mastication" is eating).  Those
  three classes are marked proxy=True or omitted, and cannot be validated from public human labels.

  python3 build_external_eval_set.py --per-class 150 --out label_tool/extern --push
"""
from __future__ import annotations
import argparse, csv, io, json, os, random, time, urllib.request, zipfile
from collections import Counter, defaultdict
from pathlib import Path

DST = "aoxo/t2a-eval-external"

# our class <- (dataset, their label).  proxy=True means the human label means something adjacent, not the
# same thing, and must be reported separately rather than folded into an accuracy number.
FSD50K_MAP = {
    "whispering": ["Whispering"],
    "breathing": ["Breathing"],
    "tapping": ["Tap", "Knock"],
    "scratching": ["Scratching_(performance_technique)"],
    "crinkling": ["Crumpling_and_crinkling"],
    "liquid": ["Liquid", "Pour", "Trickle_and_dribble", "Drip", "Splash_and_splatter"],
    "page turning": ["Writing"],
    "other sound": ["Zipper_(clothing)", "Squeak", "Typing"],
    "mouth sounds": ["Chewing_and_mastication"],          # proxy: eating, not close-mic mouth sounds
}
ESC50_MAP = {
    "breathing": ["breathing"],
    "brushing": ["brushing_teeth"],
    "liquid": ["pouring_water", "water_drops"],
    "tapping": ["door_wood_knock", "mouse_click", "keyboard_typing", "clock_tick"],
}
AUDIOSET_MAP = {
    "whispering": ["Whispering"],
    "normal speech": ["Speech", "Conversation", "Narration, monologue"],
    "silence": ["Silence"],
    "breathing": ["Breathing", "Sigh", "Gasp", "Pant", "Sniff"],
    "tapping": ["Tap", "Knock"],
    "scratching": ["Scratch", "Scratching (performance technique)"],
    "crinkling": ["Crumpling, crinkling", "Rustle"],
    "brushing": ["Toothbrush", "Electric toothbrush"],
    "liquid": ["Liquid", "Pour", "Trickle, dribble", "Drip", "Splash, splatter", "Gurgling"],
    "page turning": ["Writing"],
    "other sound": ["Squeak", "Zipper (clothing)", "Typing"],
    "moaning": ["Wail, moan", "Whimper", "Groan"],        # proxy: distress, not erotic
    "mouth sounds": ["Chewing, mastication", "Gargling"],  # proxy
}
PROXY = {("mouth sounds", "fsd50k"), ("mouth sounds", "audioset"), ("moaning", "audioset")}
NO_HUMAN_SOURCE = ["kissing"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(url: str, token: str | None = None) -> bytes:
    """Plain identity-encoded fetch: the hub client's brotli path fails on some of these files."""
    h = {"Accept-Encoding": "identity"}
    if token: h["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300).read()


def fsd50k_eval(per_class: int, rng: random.Random, token: str) -> list[dict]:
    txt = fetch("https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/labels/eval.csv", token).decode()
    rows = list(csv.DictReader(txt.splitlines()))
    log(f"FSD50K eval split: {len(rows)} clips")
    by_class: dict[str, list] = defaultdict(list)
    for r in rows:
        labs = {l.strip() for l in r["labels"].split(",")}
        for ours, theirs in FSD50K_MAP.items():
            if labs & set(theirs):
                by_class[ours].append({"fname": r["fname"], "their_labels": sorted(labs)})
                break
    out = []
    for ours, cand in by_class.items():
        rng.shuffle(cand)
        for c in cand[:per_class]:
            out.append({"uid": f"fsd50k/{c['fname']}", "t2a_class": ours, "source": "fsd50k",
                        "source_split": "eval", "their_labels": c["their_labels"],
                        "url": f"https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips/eval/{c['fname']}.wav",
                        "proxy": (ours, "fsd50k") in PROXY, "domain": "out"})
        log(f"  fsd50k {ours:14} available {len(cand):5} taken {min(len(cand), per_class)}")
    return out


def esc50(per_class: int, rng: random.Random) -> list[dict]:
    txt = fetch("https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv").decode()
    rows = list(csv.DictReader(txt.splitlines()))
    log(f"ESC-50: {len(rows)} clips")
    by_class: dict[str, list] = defaultdict(list)
    for r in rows:
        for ours, theirs in ESC50_MAP.items():
            if r["category"] in theirs: by_class[ours].append(r); break
    out = []
    for ours, cand in by_class.items():
        rng.shuffle(cand)
        for r in cand[:per_class]:
            out.append({"uid": f"esc50/{r['filename']}", "t2a_class": ours, "source": "esc50",
                        "source_split": f"fold{r['fold']}", "their_labels": [r["category"]],
                        "url": f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{r['filename']}",
                        "proxy": False, "domain": "out"})
        log(f"  esc50  {ours:14} available {len(cand):5} taken {min(len(cand), per_class)}")
    return out


def audioset_eval(per_class: int, rng: random.Random, token: str, work: Path) -> list[dict]:
    """AudioSet's eval split from the agkphysics mirror: 48 kHz FLAC with the original 3-rater labels."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        log("pyarrow missing: skipping AudioSet (pip install pyarrow)"); return []
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    shards = sorted(f for f in api.list_repo_files("agkphysics/AudioSet", repo_type="dataset")
                    if f.startswith("data/eval/") and f.endswith(".parquet"))
    log(f"AudioSet eval shards: {len(shards)}")
    want = {t: ours for ours, ts in AUDIOSET_MAP.items() for t in ts}
    got: dict[str, int] = Counter(); out = []
    audio_dir = work / "audioset"; audio_dir.mkdir(parents=True, exist_ok=True)
    for sh in shards:
        if all(got[c] >= per_class for c in AUDIOSET_MAP): break
        p = hf_hub_download("agkphysics/AudioSet", sh, repo_type="dataset", cache_dir=str(work / "cache"))
        try:
            t = pq.read_table(p, columns=["video_id", "human_labels", "audio"])
            for row in t.to_pylist():
                labs = list(row.get("human_labels") or [])
                hit = next((want[l] for l in labs if l in want), None)
                if not hit or got[hit] >= per_class: continue
                aud = row.get("audio") or {}
                data = aud.get("bytes")
                if not data: continue
                name = f"{hit.replace(' ', '_')}/{row['video_id']}.flac"
                dest = audio_dir / name; dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                out.append({"uid": f"audioset/{row['video_id']}", "t2a_class": hit, "source": "audioset",
                            "source_split": "eval", "their_labels": labs, "local": str(dest),
                            "proxy": (hit, "audioset") in PROXY, "domain": "out"})
                got[hit] += 1
        finally:
            try: os.remove(os.path.realpath(p))
            except OSError: pass
        log(f"  after {sh}: " + ", ".join(f"{k}={got[k]}" for k in sorted(got)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=150, help="per class per source")
    ap.add_argument("--sources", default="fsd50k,esc50,audioset")
    ap.add_argument("--push", action="store_true", help="publish clips + manifest to the eval repo")
    ap.add_argument("--seed", type=int, default=20260925)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed); token = os.environ["HF_TOKEN"]
    picked: list[dict] = []
    todo = {s.strip() for s in a.sources.split(",")}
    if "fsd50k" in todo: picked += fsd50k_eval(a.per_class, rng, token)
    if "esc50" in todo: picked += esc50(a.per_class, rng)
    if "audioset" in todo: picked += audioset_eval(a.per_class, rng, token, a.out)

    manifest = a.out / "external_eval_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in picked))
    per_class = Counter(r["t2a_class"] for r in picked)
    stats = {"clips": len(picked), "per_class": dict(per_class),
             "by_source": dict(Counter(r["source"] for r in picked)),
             "proxy_classes": sorted({r["t2a_class"] for r in picked if r["proxy"]}),
             "no_human_source": NO_HUMAN_SOURCE, "seed": a.seed}
    json.dump(stats, open(a.out / "external_eval_stats.json", "w"), indent=2)
    log(f"{len(picked)} clips -> {manifest}")
    log(json.dumps(stats["per_class"]))
    log(f"proxy (report separately): {stats['proxy_classes']}   no human source at all: {NO_HUMAN_SOURCE}")

    if a.push:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(DST, repo_type="dataset", private=False, exist_ok=True)
        api.upload_file(path_or_fileobj=str(manifest), path_in_repo="external_eval_manifest.jsonl",
                        repo_id=DST, repo_type="dataset", commit_message=f"{len(picked)} human-labelled eval clips")
        api.upload_file(path_or_fileobj=str(a.out / "external_eval_stats.json"), path_in_repo="stats.json",
                        repo_id=DST, repo_type="dataset", commit_message="eval set stats")
        log(f"pushed manifest to {DST} (clips stay at their sources except AudioSet, which is bundled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
