#!/usr/bin/env python3
"""Build the in-domain listening queue a human labels to gate CLAP v7 (the 90%-per-class bar).

Sources, both unseen by every model we train:
  * vocal classes: humaneval/human_eval_manifest.jsonl -- 120 reserved creators the training builders refuse
  * physical classes: YouTube windows from the held-out eval split (video-level split, never trained on)

The stratum (what a model thinks the clip is) chooses the sample and is written ONLY to key.jsonl, never to the
page: the listener must not be anchored by a model's guess. Clips are interleaved across strata so a partial
session is still balanced. Audio ships as a few Opus sprite files (one request each) with per-clip offsets.

  python build_listen_queue.py --per-class 30 --out /root/t2a/listen
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path

VOCAL = {"breathing": "breathing", "kissing": "oral sounds", "mouth sounds": "oral sounds", "moaning": "moaning",
         "whispering": "whispering", "normal speech": "normal speech", "silence": "silence", "__uniform__": "uniform"}
VOCAL_N = {"breathing": 1, "oral sounds": 1, "moaning": 1, "whispering": 0.6, "normal speech": 0.6, "silence": 0.6, "uniform": 1.3}
PHYSICAL = ["tapping", "crinkling", "scratching", "brushing", "liquid", "microphone touching", "sticky",
            "fabric rustling", "paper rustling", "cutting"]
SR = 24_000
GAP_S = 0.4


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cut(url: str, start: float, dur: float, out: Path) -> bool:
    for i in range(3):
        try:
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}", "-i", url,
                            "-ac", "1", "-ar", str(SR), "-af", "loudnorm=I=-23:TP=-2", str(out)], capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            continue                                     # a stalled remote read: retry, never hang the build
        if r.returncode == 0 and out.exists() and out.stat().st_size > 20000: return True
        time.sleep(5 * (i + 1))
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--humaneval", type=Path, default=Path("/root/t2a/humaneval/human_eval_manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("/root/t2a/listen"))
    ap.add_argument("--sprite-clips", type=int, default=150)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    rng = random.Random(20260928)
    a.out.mkdir(parents=True, exist_ok=True); wav = a.out / "wav"; wav.mkdir(exist_ok=True)

    # ---- vocal strata from reserved creators; at most 2 clips per creator per stratum ----
    by = defaultdict(list)
    for l in open(a.humaneval):
        r = json.loads(l); s = VOCAL.get(r["stratum"])
        if s: by[s].append(r)
    picks = defaultdict(list)
    for s, rows in by.items():
        rng.shuffle(rows); per_c = Counter(); want = round(a.per_class * VOCAL_N[s])
        for r in rows:
            if len(picks[s]) >= want: break
            if per_c[r["creator"]] >= 2: continue
            per_c[r["creator"]] += 1
            ms = int(r["uid"].rsplit("_", 1)[1])
            picks[s].append({"stratum": s, "uid": r["uid"], "creator": r["creator"],
                             "url": f"https://huggingface.co/datasets/{r['repo']}/resolve/main/{r['source']}",
                             "start": ms / 1000 - 0.5, "dur": 6.0})
    # ---- physical strata from held-out YouTube videos; at most 2 windows per video ----
    yt = [json.loads(l) for l in open(hf_hub_download("aoxo/clap-ft-data", "yt_windows/windows.jsonl", repo_type="dataset"))]
    yt = [r for r in yt if r.get("split") == "eval" and r["label"] in PHYSICAL]
    rng.shuffle(yt); per_v = Counter()
    for r in yt:
        s = r["label"]
        if len(picks[s]) >= a.per_class or per_v[(s, r["source"])] >= 2: continue
        per_v[(s, r["source"])] += 1
        vid = r["source"].split(":", 1)[1]
        picks[s].append({"stratum": s, "uid": r["uid"], "creator": r["source"],
                         "url": f"https://huggingface.co/datasets/{r['repo']}/resolve/main/audio/{vid}.flac",
                         "start": r["start"] - 0.5, "dur": 5.0})
    log("sampled: " + ", ".join(f"{s}={len(v)}" for s, v in sorted(picks.items())))

    # interleave strata so any prefix of the queue is balanced
    order = [p for grp in zip_longest(*[rng.sample(v, len(v)) for v in picks.values()]) for p in grp if p]
    for i, p in enumerate(order): p["id"] = f"c{i:04d}"

    def job(p):
        time.sleep(0.4)                                  # shared Hub quota
        return p, cut(p["url"], p["start"], p["dur"], wav / f"{p['id']}.wav")
    ok = []
    with ThreadPoolExecutor(8) as ex:
        for i, (p, good) in enumerate(ex.map(job, order)):
            if good: ok.append(p)
            if i % 50 == 0: log(f"  cut {i}/{len(order)}, {len(ok)} ok")
    log(f"{len(ok)}/{len(order)} clips cut")

    # ---- sprites: clips back to back with a short gap; offsets go to the page ----
    clips, sprite_i = [], 0
    for b in range(0, len(ok), a.sprite_clips):
        chunk = ok[b:b + a.sprite_clips]; lst = a.out / f"sprite_{sprite_i}.txt"; t = 0.0
        silence = a.out / "gap.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=mono", "-t", str(GAP_S), str(silence)], check=True)
        with lst.open("w") as fh:
            for p in chunk:
                d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                          str(wav / f"{p['id']}.wav")], capture_output=True, text=True).stdout.strip())
                fh.write(f"file '{wav / (p['id'] + '.wav')}'\nfile '{silence}'\n")
                clips.append({"id": p["id"], "sprite": f"sprite_{sprite_i}.ogg", "t": round(t, 3), "dur": round(d, 3)})
                t += d + GAP_S
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:a", "libopus",
                        "-b:a", "40k", str(a.out / f"sprite_{sprite_i}.ogg")], check=True)
        sprite_i += 1
    (a.out / "clips.json").write_text(json.dumps(clips))
    # the key stays off the page: stratum is a model's guess and must not anchor the listener
    (a.out / "key.jsonl").write_text("".join(json.dumps({k: p[k] for k in ("id", "stratum", "uid", "creator", "start", "dur")}) + "\n" for p in ok))
    log(f"LISTEN_DONE {len(clips)} clips in {sprite_i} sprites -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
