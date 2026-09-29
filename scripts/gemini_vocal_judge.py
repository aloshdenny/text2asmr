#!/usr/bin/env python3
"""Second opinion on the vocal classes: Gemini judges clips Qwen3-Omni already labelled, plus a human-labelled
calibration set, so fuse_judges.py can estimate how precise the Qwen labels really are.

  calib    : human-labelled external clips (breathing / mouth sounds / whispering / other) -> Gemini's vocal
             confusion matrix, the anchor. Moaning has no human-labelled source anywhere, so it stays unanchored.
  indomain : --per-class clips per Qwen class, drawn from many recordings (cap --per-recording each, so no
             creator dominates), cut [start-1 s, +8 s] exactly as Qwen saw them, judged, deleted.

Shares gemini_clean_yt's spend ledger: one $1000 cap across every Gemini job. Paced for the shared Hub quota.

  python gemini_vocal_judge.py --mode calib --out /root/t2a/gvocal
  python gemini_vocal_judge.py --mode indomain --per-class 3000 --out /root/t2a/gvocal
"""
from __future__ import annotations
import argparse, base64, json, os, random, subprocess, sys, tempfile, threading, time, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemini_clean_yt import Ledger, MODEL

CLASSES = ["breathing", "oral sounds", "moaning", "whispering", "normal speech"]
QWEN_MAP = {"breathing": "breathing", "kissing": "oral sounds", "mouth sounds": "oral sounds", "licking": "oral sounds",
            "moaning": "moaning", "whispering": "whispering", "normal speech": "normal speech"}
HUMAN_MAP = {"breathing": "breathing", "mouth sounds": "oral sounds", "whispering": "whispering", "other sound": "other"}
PROMPT = ("This is a short clip from an ASMR recording. Which ONE best describes the main sound in it? "
          "breathing; oral sounds (kissing, licking, lip or mouth sounds); moaning; whispering (whispered words); "
          "normal speech (voiced talking); none of these. Answer with the label only.")


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def parse(t: str) -> str | None:
    t = (t or "").lower()
    for k, v in (("none", "other"), ("oral", "oral sounds"), ("kiss", "oral sounds"), ("lick", "oral sounds"),
                 ("mouth", "oral sounds"), ("normal", "normal speech"), ("whisper", "whispering"), ("moan", "moaning"),
                 ("breath", "breathing")):
        if k in t: return v
    return None


def judge(wav: Path, key: str) -> dict:
    body = {"model": MODEL, "temperature": 0, "max_tokens": 2000, "usage": {"include": True},
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav.read_bytes()).decode(), "format": "wav"}}]}]}
    for i in range(7):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=240).read())
            if "error" in d: raise RuntimeError(str(d["error"])[:120])
            u = d.get("usage") or {}
            cost = float(u.get("cost") or 0) + (float((u.get("cost_details") or {}).get("upstream_inference_cost") or 0) if u.get("is_byok") else 0)
            txt = d["choices"][0]["message"].get("content") or ""
            return {"pred": parse(txt), "raw": txt.strip()[:60], "gen_id": d.get("id"), "cost": cost}
        except Exception as e:
            err = getattr(e, "read", lambda: b"")() or str(e).encode()
            if any(s in err for s in (b"429", b"500", b"502", b"503", b"504", b"520", b"529", b"timed out", b"aborted")) and i < 6:
                time.sleep(min(90, 5 * 2 ** i)); continue
            return {"pred": None, "error": err[:120].decode(errors="replace"), "cost": 0.0}


def cut(src: str, start: float, dur: float, out: Path) -> bool:
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}", "-i", src,
                        "-ac", "1", "-ar", "16000", str(out)], capture_output=True)
    return r.returncode == 0 and out.exists() and out.stat().st_size > 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calib", "indomain"], required=True)
    ap.add_argument("--out", type=Path, default=Path("/root/t2a/gvocal"))
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--per-recording", type=int, default=6)
    ap.add_argument("--classes", default=",".join(CLASSES), help="Qwen classes to sample (e.g. breathing,oral sounds)")
    ap.add_argument("--ledger", type=Path, default=Path("/root/t2a/gclean/ledger.json"))
    ap.add_argument("--cap", type=float, default=1000.0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--dl-workers", type=int, default=3, help="recordings downloading at once")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, hf_hub_download
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from text2asmr.io_guard import JsonlSink
    a.out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(a.ledger, a.cap)
    key = os.environ["OPENROUTER_API_KEY_GS"]
    sink = JsonlSink(a.out / f"{a.mode}.jsonl", key="uid")
    # a row without a prediction (provider timeout) is not done: judge it again rather than skip it forever
    ok_uids = set()
    if (a.out / f"{a.mode}.jsonl").exists():
        for l in open(a.out / f"{a.mode}.jsonl"):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("pred") is not None: ok_uids.add(r["uid"])
    sink.seen = ok_uids
    tmp = Path(tempfile.mkdtemp(prefix="gvocal_", dir=str(a.out)))
    n, lock = [0], threading.Lock()

    def one(uid, wav, meta):
        if ledger.over(): return
        j = judge(wav, key); wav.unlink(missing_ok=True); ledger.add(j["cost"])
        with lock:
            sink.write(dict(meta, uid=uid, **j)); n[0] += 1
            if n[0] % 250 == 0: log(f"  {n[0]} judged, ledger ${ledger.spent:.2f}")

    if a.mode == "calib":
        rows = [json.loads(l) for l in open(hf_hub_download("aoxo/t2a-eval-external", "external_eval_manifest.jsonl", repo_type="dataset"))]
        rows = [r for r in rows if r["t2a_class"] in HUMAN_MAP and r["uid"] not in sink.seen]
        log(f"calibration: {Counter(r['t2a_class'] for r in rows)}")
        with ThreadPoolExecutor(a.concurrency) as ex:
            for r in rows:
                w = tmp / (r["uid"].replace("/", "_") + ".wav")
                time.sleep(0.4)
                if cut(r["url"], 0.0, 8.0, w): ex.submit(one, r["uid"], w, {"truth": HUMAN_MAP[r["t2a_class"]]})
    else:
        # recording-first sampling: every clip costs a full recording download (~15 MB), so drawing clips
        # uniformly gave ~2 clips per download. Pick recordings at random, then take up to --per-recording
        # clips of the wanted classes from each: the same creator spread, ~4x fewer downloads.
        want_classes = set(a.classes.split(","))
        rng = random.Random(0); pool, seen_n = defaultdict(lambda: defaultdict(list)), Counter()
        for repo in ("aoxo/t2a-mommy", "aoxo/t2a-daddy"):
            for f in HfApi().list_repo_files(repo, repo_type="dataset"):
                if not (f.startswith("labels/qwen3omni") and f.endswith(".jsonl")): continue
                for l in open(hf_hub_download(repo, f, repo_type="dataset")):
                    try: r = json.loads(l)
                    except Exception: continue
                    c = QWEN_MAP.get(r.get("label"))
                    if c not in want_classes: continue
                    seen_n[c] += 1
                    pool[(repo, r["uid"].rsplit("_", 1)[0])][c].append(r["uid"])
        recs = sorted(pool); rng.shuffle(recs)
        by_src, taken = defaultdict(list), Counter()
        for k in recs:
            if all(taken[c] >= a.per_class for c in want_classes): break
            for c, uids in pool[k].items():
                room = a.per_class - taken[c]
                if room <= 0: continue
                pick = rng.sample(uids, min(len(uids), a.per_recording, room))
                by_src[k] += [(u, c) for u in pick]; taken[c] += len(pick)
        log(f"in-domain sample: {dict(taken)} from {len(by_src)} recordings (pool sizes {dict(seen_n)})")
        # one recording per download worker; each judges its clips on the shared Gemini pool. Sequential
        # downloads made this ~550 clips/h (27 h for the full pass); three in flight, still Hub-paced.
        judge_pool = ThreadPoolExecutor(a.concurrency)

        def do_recording(k):
            (repo, src), items = k
            if ledger.over(): return
            local = None
            for i in range(5):
                try:
                    time.sleep(0.5); local = hf_hub_download(repo, src, repo_type="dataset", cache_dir=str(a.out / "cache")); break
                except Exception:
                    time.sleep(20 * 2 ** i)
            if not local: return
            futs = []
            for uid, c in items:
                w = tmp / (uid.replace("/", "_") + ".wav")
                if cut(local, int(uid.rsplit("_", 1)[1]) / 1000 - 1.0, 8.0, w):
                    futs.append(judge_pool.submit(one, uid, w, {"qwen": c, "repo": repo}))
            for fu in futs: fu.result()
            try: os.remove(os.path.realpath(local))
            except OSError: pass
        todo = [(k, [(u, c) for u, c in v if u not in sink.seen]) for k, v in by_src.items()]
        todo = [t for t in todo if t[1]]
        log(f"{sum(len(v) for _, v in todo)} clips left in {len(todo)} recordings")
        with ThreadPoolExecutor(a.dl_workers) as dl:
            list(dl.map(do_recording, todo))
        judge_pool.shutdown(wait=True)
    sink.close()
    log(f"DONE {a.mode}: {n[0]} judged, ledger ${ledger.spent:.2f} / ${a.cap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
