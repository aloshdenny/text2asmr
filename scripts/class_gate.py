#!/usr/bin/env python3
"""Admission test for ontology classes that come from YouTube chapter titles.

A chapter called "sticky" tells us what the creator *meant* to record, not what a 4 s window of it sounds
like. Before a class goes into CLAP v7 we ask: when independent audio models hear a window from that
chapter, with no hint of the title, do they pick the same class out of the full list?

  * per-class hit rate (judge answer == chapter label)  -> is the chapter label a usable training target?
  * inter-judge agreement per class                      -> is the class perceptually distinct at all?

Judges are credit-billed OpenRouter models from three families (no BYOK: the key's own spend limit is the
budget). Sampling takes at most --per-video windows from any one video so a single creator cannot carry a
class.

  python3 class_gate.py --stage cut,judge,report --work /root/t2a/gate
"""
from __future__ import annotations
import argparse, base64, json, os, random, subprocess, sys, threading, time, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text2asmr.io_guard import JsonlSink, preflight, safe_name

CLASSES = ["tapping", "brushing", "scratching", "liquid", "microphone touching", "crinkling", "sticky",
           "fabric rustling", "paper rustling", "hand movements", "cutting", "typing", "writing", "page turning"]
JUDGES = ["xiaomi/mimo-v2.6-flash", "perceptron/perceptron-mk1.5", "qwen/qwen3.8-omni-flash"]
PROMPT = ("This is a 4-second clip from an ASMR video. Which ONE of these sounds is it mainly? "
          + "; ".join(CLASSES) + "; none of these. Answer with the label only, nothing else.")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse(t: str) -> str | None:
    t = (t or "").lower().strip().strip(".")
    # longest names first so "paper rustling" is not read as a bare "rustling" hit elsewhere
    for c in sorted(CLASSES + ["none of these"], key=len, reverse=True):
        if c in t: return c
    return None


def stage_cut(a):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet buffers whole files in RAM -> OOM on 1 GB
    from huggingface_hub import hf_hub_download
    rows = [json.loads(l) for l in open(hf_hub_download("aoxo/clap-ft-data", "yt_windows/windows.jsonl",
                                                        repo_type="dataset"))]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    per_vid, picked = Counter(), defaultdict(list)
    for r in rows:
        c = r.get("label")
        if c in CLASSES and len(picked[c]) < a.per_class and per_vid[(c, r["source"])] < a.per_video:
            picked[c].append(r); per_vid[(c, r["source"])] += 1
    sel = [r for c in CLASSES for r in picked[c]]
    log("sampled: " + ", ".join(f"{c}={len(picked[c])}" for c in CLASSES))
    del rows, picked                                    # the droplet has 1 GB; 69k dicts is most of it

    wav_dir = a.work / "wav"; wav_dir.mkdir(parents=True, exist_ok=True)
    # ffmpeg seeks the remote FLAC with HTTP range requests: ~19 s per clip but no download, no disk and
    # ~20 MB RAM, so 8 in parallel beats fetching whole 200 MB sources to keep two 4 s windows from each.
    def cut(r):
        w = wav_dir / (safe_name(r["uid"]) + ".wav")
        if not w.exists():
            url = f"https://huggingface.co/datasets/{r['repo']}/resolve/main/audio/{r['source'].split(':', 1)[1]}.flac"
            tmp = w.with_suffix(".part.wav")
            p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{r['start']:.3f}", "-t", f"{r['dur']:.3f}",
                                "-i", url, "-ac", "1", "-ar", "16000", str(tmp)], capture_output=True, text=True)
            if p.returncode or not tmp.exists() or tmp.stat().st_size < 1000:
                return None, f"{r['uid']}: {p.stderr.strip()[:100]}"
            tmp.rename(w)                               # atomic: a killed run never leaves a truncated clip
        return {"uid": r["uid"], "label": r["label"], "source": r["source"], "chapter": r.get("chapter"),
                "wav": str(w)}, None
    clips = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for i, (c, err) in enumerate(ex.map(cut, sel)):
            if c: clips.append(c)
            if err: log(f"  skip {err}")
            if i % 40 == 0: log(f"  cut {i}/{len(sel)} clips, {len(clips)} ok")
    (a.work / "clips.json").write_text(json.dumps(clips))
    log(f"{len(clips)} clips -> {a.work / 'clips.json'}")


def stage_judge(a):
    clips = json.loads((a.work / "clips.json").read_text())
    key = os.environ["OPENROUTER_API_KEY_GS"]
    for model in a.judges:
        out = a.work / f"judge_{safe_name(model)}.jsonl"
        preflight(out, note=model)
        done = set()
        if out.exists():
            for l in out.open():
                try: r = json.loads(l)
                except Exception: continue
                if r.get("pred") is not None: done.add(r["uid"])
        todo = [c for c in clips if c["uid"] not in done]
        log(f"{model}: {len(todo)} to judge ({len(done)} already), cap ${a.max_cost}")
        spent = [0.0]; lock = threading.Lock()
        sink = JsonlSink(out, key="uid")

        def one(c):
            if spent[0] >= a.max_cost: return {"uid": c["uid"], "pred": None, "error": "budget"}
            body = {"model": model, "temperature": 0, "max_tokens": 400,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "input_audio", "input_audio": {
                            "data": base64.b64encode(open(c["wav"], "rb").read()).decode(), "format": "wav"}}]}],
                    "provider": {"allow_fallbacks": True}}
            try:
                req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                             "HTTP-Referer": "https://github.com/aloshdenny/text2asmr"})
                d = json.loads(urllib.request.urlopen(req, timeout=180).read())
                if "error" in d: return {"uid": c["uid"], "pred": None, "error": str(d["error"])[:160]}
                u = d.get("usage") or {}
                with lock: spent[0] += float(u.get("cost") or 0.0)
                txt = d["choices"][0]["message"].get("content") or ""
                return {"uid": c["uid"], "pred": parse(txt), "raw": txt.strip()[:120], "gen_id": d.get("id"),
                        "cost": u.get("cost")}
            except Exception as e:
                err = getattr(e, "read", lambda: b"")()
                return {"uid": c["uid"], "pred": None, "error": f"{type(e).__name__} {err[:160]!r}"}

        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            for i, r in enumerate(ex.map(one, todo)):
                sink.write(r)                            # paid rows hit disk before the next result
                if i % 50 == 0: log(f"  {model} {i}/{len(todo)} ${spent[0]:.3f} pred={r.get('pred')} err={r.get('error','')[:60]}")
        sink.close()
        log(f"{model}: done, ${spent[0]:.3f}")


def stage_report(a):
    clips = {c["uid"]: c for c in json.loads((a.work / "clips.json").read_text())}
    J = {}
    for p in sorted(a.work.glob("judge_*.jsonl")):
        d = {}
        for l in p.open():
            try: r = json.loads(l)
            except Exception: continue
            if r.get("pred"): d[r["uid"]] = r["pred"]
        J[p.stem.replace("judge_", "")] = d
    lines = [f"judges: " + ", ".join(f"{k}({len(v)})" for k, v in J.items()), "",
             f"{'class':20} {'n':>3} " + " ".join(f"{k[:14]:>14}" for k in J) + "   judges-agree  verdict"]
    verdicts = {}
    for c in CLASSES:
        uids = [u for u, x in clips.items() if x["label"] == c]
        hits = []
        for k, d in J.items():
            got = [u for u in uids if u in d]
            hits.append(sum(d[u] == c for u in got) / max(len(got), 1))
        pair = []
        for x, y in combinations(J, 2):
            both = [u for u in uids if u in J[x] and u in J[y]]
            if both: pair.append(sum(J[x][u] == J[y][u] for u in both) / len(both))
        agree = sum(pair) / max(len(pair), 1)
        mean_hit = sum(hits) / max(len(hits), 1)
        # admit when the chapter label is heard more often than not by the judges on average; chance on a
        # 15-way question is ~7%, so even 0.35 means the label carries real signal
        v = "ADMIT" if mean_hit >= 0.5 else "WEAK" if mean_hit >= 0.3 else "REJECT"
        verdicts[c] = {"n": len(uids), "hit": hits, "mean_hit": mean_hit, "judge_agree": agree, "verdict": v}
        lines.append(f"{c:20} {len(uids):>3} " + " ".join(f"{h:>14.0%}" for h in hits) + f"   {agree:>11.0%}  {v}")
    # where do misses go? the confusion is what tells us which classes to merge
    lines += ["", "most common wrong answers (all judges pooled):"]
    for c in CLASSES:
        wrong = Counter(d[u] for d in J.values() for u, x in clips.items() if x["label"] == c and u in d and d[u] != c)
        if wrong: lines.append(f"  {c:20} -> " + ", ".join(f"{k} {n}" for k, n in wrong.most_common(3)))
    txt = "\n".join(lines); print(txt)
    (a.work / "gate_report.txt").write_text(txt)
    (a.work / "gate_verdicts.json").write_text(json.dumps(verdicts, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/root/t2a/gate"))
    ap.add_argument("--stage", default="cut,judge,report")
    ap.add_argument("--judges", default=",".join(JUDGES))
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--per-video", type=int, default=2)
    ap.add_argument("--max-cost", type=float, default=1.5, help="USD cap per judge")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    a.judges = [j for j in a.judges.split(",") if j]
    a.work.mkdir(parents=True, exist_ok=True)
    for s in a.stage.split(","):
        {"cut": stage_cut, "judge": stage_judge, "report": stage_report}[s](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
