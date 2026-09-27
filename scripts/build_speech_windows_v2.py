#!/usr/bin/env python3
"""Generator v2 training windows: packed speech with explicit pauses and vocal events (docs/GENERATOR_V2.md).

Per recording (word alignment + Qwen3-Omni labels on its gaps):
  * consecutive phrases are packed, *with the real pauses between them*, into windows whose target length is
    drawn uniformly from [--min-s, --max-s], so short and long windows are both well covered (the corpus'
    median phrase is 3 s; unpacked, a model would rarely see 15-20 s)
  * every gap inside a window becomes explicit text: a vocal-event tag where Qwen labelled the gap
    ([breathing], [oral sounds], [moaning]) or a pause tag rounded to 0.5 s. The model is never left to
    invent a pause or a breath on its own
  * a gap longer than --max-gap-s ends the window (that is trigger territory, not speech)
  * a gap labelled as speech (whispering / normal speech) means the aligner missed words there: the transcript
    would be incomplete, so the window is dropped rather than trained on wrong text
  * each window records the --prompt-s of audio before it as the continuation prompt, when the recording has it

Balance: at most --creator-cap-h hours per creator, and --hours total split by corpus weight.

  python3 build_speech_windows_v2.py --hours 2000 --out speech_windows_v2.jsonl
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text2asmr.data.segment import split_alignment
from text2asmr.io_guard import JsonlSink, preflight

VOCAL = {"breathing": "breathing", "moaning": "moaning", "kissing": "oral sounds", "mouth sounds": "oral sounds",
         "licking": "oral sounds", "oral sounds": "oral sounds"}
SPEECHY = {"whispering", "normal speech"}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


_pace_lock = __import__("threading").Lock(); _pace_next = [0.0]
PACE_S = 0.25                                           # one Hub file every 0.25 s across all threads


def pace():
    with _pace_lock:
        now = time.time(); wait = _pace_next[0] - now
        _pace_next[0] = max(now, _pace_next[0]) + PACE_S
    if wait > 0: time.sleep(wait)


def load_labels(repo: str) -> dict[str, dict[int, str]]:
    """source -> {gap start ms: label}, from every Qwen3-Omni ledger in the repo."""
    from huggingface_hub import HfApi, hf_hub_download
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for f in HfApi().list_repo_files(repo, repo_type="dataset"):
        if not (f.startswith("labels/qwen3omni") and f.endswith(".jsonl")): continue
        for line in open(hf_hub_download(repo, f, repo_type="dataset")):
            try: r = json.loads(line)
            except Exception: continue
            src, ms = r["uid"].rsplit("_", 1)
            out[src][int(ms)] = r["label"]
    return out


def looped(text: str, max_repeat: int) -> bool:
    """True if any word repeats more than max_repeat times in a row (ignoring case and punctuation)."""
    toks = ["".join(ch for ch in t.lower() if ch.isalnum()) for t in text.split() if not t.startswith("[")]
    run = 1
    for x, y in zip(toks, toks[1:]):
        run = run + 1 if x and x == y else 1
        if run > max_repeat: return True
    return False


dropped: Counter = Counter()


def windows_for(src: str, align: list[dict], labels: dict[int, str], a, rng) -> list[dict]:
    spans = split_alignment(align, src)
    phrases = [s for s in spans if s.kind == "speech"]
    if not phrases: return []
    lab_starts = sorted(labels)

    def gap_tag(g0: float, g1: float) -> str | None:
        """Label of the gap [g0, g1): any Qwen clip starting inside it. None means 'drop this window'."""
        hits = [labels[m] for m in lab_starts if g0 * 1000 - 50 <= m < g1 * 1000]
        if any(h in SPEECHY for h in hits): return None
        vocal = [VOCAL[h] for h in hits if h in VOCAL]
        if vocal: return f"[{Counter(vocal).most_common(1)[0][0]}]"
        return f"[pause {max(0.5, round((g1 - g0) * 2) / 2):.1f}s]"

    out, i = [], 0
    while i < len(phrases):
        target = rng.uniform(a.min_s, a.max_s)
        start = phrases[i].start; parts = [phrases[i].text.strip()]; end = phrases[i].end; j = i + 1; ok = True
        while j < len(phrases):
            g0, g1 = phrases[j - 1].end, phrases[j].start
            if g1 - g0 > a.max_gap_s or phrases[j].end - start > target: break
            if g1 - g0 >= 0.7:
                t = gap_tag(g0, g1)
                if t is None: ok = False; break
                parts.append(t)
            parts.append(phrases[j].text.strip()); end = phrases[j].end; j += 1
        if ok and end - start >= a.min_s and end - start <= a.max_s:
            text = " ".join(p for p in parts if p)
            if looped(text, a.max_repeat):
                # Whisper transcribes moans and breaths as "uh, uh, uh, uh...": a looping target teaches the TTS
                # to loop, which is the exact failure v2 exists to avoid. Guessing which event it was is worse
                # than dropping the window.
                dropped["looped transcript"] += 1; i = max(j, i + 1); continue
            if len(text.split()) <= a.max_words:
                out.append({"uid": f"{src}_{int(start * 1000):09d}", "source": src, "start": round(start, 3),
                            "end": round(end, 3), "dur": round(end - start, 3), "text": text,
                            "prompt_start": round(start - a.prompt_s, 3) if start >= a.prompt_s else None,
                            "prompt_end": round(start, 3) if start >= a.prompt_s else None})
        i = max(j, i + 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="aoxo/t2a-mommy,aoxo/t2a-daddy")
    ap.add_argument("--hours", type=float, default=2000.0, help="total window hours, split evenly across repos")
    ap.add_argument("--creator-cap-h", type=float, default=10.0)
    ap.add_argument("--min-s", type=float, default=2.0); ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--max-gap-s", type=float, default=6.0); ap.add_argument("--prompt-s", type=float, default=4.0)
    ap.add_argument("--max-words", type=int, default=55)
    ap.add_argument("--max-repeat", type=int, default=3, help="drop windows with a word repeated more than this in a row")
    ap.add_argument("--limit-files", type=int, default=0, help="debug: stop after N recordings per repo")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    preflight(a.out, note="speech windows v2")
    sink = JsonlSink(a.out, key="uid")
    repos = a.repos.split(","); per_repo_s = a.hours * 3600 / len(repos)
    rng = random.Random(0)
    tag_counts, stats = Counter(), Counter()
    for repo in repos:
        labels = load_labels(repo)
        log(f"{repo}: labels for {len(labels)} recordings")
        files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".m4a.json")]
        rng.shuffle(files)                              # random order = no creator gets there first
        if a.limit_files: files = files[:a.limit_files]
        got_s, per_creator = 0.0, Counter()

        def fetch(f):
            # The Hub allows 5,000 resolves per 5 min for the whole account, shared with the label pods and the
            # droplet. Unpaced, 16 threads blew through it in two minutes and every failure was silently
            # counted as an empty alignment. Pace, retry 429s, and report failures separately.
            for i in range(7):
                pace()
                try: return f, json.load(open(hf_hub_download(repo, f, repo_type="dataset")))
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e) or "LocalEntryNotFound" in type(e).__name__:
                        time.sleep(min(300, 20 * 2 ** i)); continue
                    return f, "error"
            return f, "error"
        with ThreadPoolExecutor(a.workers) as ex:
            for f, align in ex.map(fetch, files):
                if got_s >= per_repo_s: break
                src = f[:-len(".json")]; creator = src.split("/")[0]
                if align == "error": stats["fetch failed"] += 1; continue
                if not align: stats["empty alignment"] += 1; continue
                if per_creator[creator] >= a.creator_cap_h * 3600: stats["creator cap"] += 1; continue
                for w in windows_for(src, align, labels.get(src, {}), a, rng):
                    if per_creator[creator] >= a.creator_cap_h * 3600: break
                    w.update(repo=repo, creator=creator)
                    sink.write(w); got_s += w["dur"]; per_creator[creator] += w["dur"]; stats["windows"] += 1
                    for t in w["text"].split("["):
                        if "]" in t: tag_counts[t.split("]")[0].split(" ")[0]] += 1
                stats["recordings"] += 1
        log(f"{repo}: {got_s / 3600:.0f} h from {len(per_creator)} creators")
    sink.close()
    log(f"stats: {dict(stats)}; dropped: {dict(dropped)}")
    log("inline tags: " + ", ".join(f"{k}={v}" for k, v in tag_counts.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
