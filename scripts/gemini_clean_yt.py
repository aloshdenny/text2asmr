#!/usr/bin/env python3
"""Filter YouTube chapter windows with Gemini: keep a window only when Gemini hears its chapter's class.

Why only some classes: on human-labelled FSD50K/ESC-50 clips (class_gate.py --stage cutext) Gemini 3.1 Pro
recognised brushing 88%, crinkling 80%, liquid 75%, tapping 57%, but scratching 30% and page turning 0%.
A judge only filters where it can hear; elsewhere it would throw away good windows at random.

Spend: Gemini routes through the org's BYOK AI Studio key, so OpenRouter reports cost=0 and the key's own
limit does not apply. The cap here is the only brake, so it is persisted in a ledger file -- a restart must
never reset it.

Each video is downloaded once and all its target windows are cut from the local copy, then deleted. The
droplet has 1 GB RAM and ~15 GB disk, so everything streams.

  python3 gemini_clean_yt.py --cap 1000 --out /root/t2a/gclean
"""
from __future__ import annotations
import argparse, base64, json, os, queue, random, subprocess, sys, threading, time, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from class_gate import PROMPT, parse                    # the exact prompt the calibration scored
from text2asmr.io_guard import JsonlSink, preflight, safe_name

MODEL = "google/gemini-3.1-pro-preview"
TARGETS = ["brushing", "liquid", "crinkling", "tapping"]


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


class Ledger:
    """Cumulative spend across every run, on disk. The cap is checked against this, not a per-run counter."""
    def __init__(self, path: Path, cap: float):
        self.path, self.cap, self.lock = path, cap, threading.Lock()
        self.spent = json.loads(path.read_text())["spent"] if path.exists() else 0.0

    def add(self, x: float):
        with self.lock:
            self.spent += x
            tmp = self.path.with_suffix(".tmp"); tmp.write_text(json.dumps({"spent": self.spent, "cap": self.cap}))
            tmp.replace(self.path)

    def over(self) -> bool: return self.spent >= self.cap


def judge(wav: Path, key: str) -> dict:
    body = {"model": MODEL, "temperature": 0, "max_tokens": 2000, "usage": {"include": True},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav.read_bytes()).decode(),
                                                        "format": "wav"}}]}]}
    for attempt in range(7):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                         "HTTP-Referer": "https://github.com/aloshdenny/text2asmr"})
            d = json.loads(urllib.request.urlopen(req, timeout=240).read())
            if "error" in d: raise RuntimeError(str(d["error"])[:160])
            u = d.get("usage") or {}
            cost = float(u.get("cost") or 0.0) + (float((u.get("cost_details") or {}).get("upstream_inference_cost") or 0.0)
                                                  if u.get("is_byok") else 0.0)
            txt = d["choices"][0]["message"].get("content") or ""
            return {"pred": parse(txt), "raw": txt.strip()[:80], "gen_id": d.get("id"), "cost": cost}
        except Exception as e:
            err = getattr(e, "read", lambda: b"")() or str(e).encode()
            if any(s in err for s in (b"429", b"500", b"502", b"503", b"520", b"529", b"timed out")) and attempt < 6:
                time.sleep(min(90, 5 * 2 ** attempt) + random.random()); continue
            return {"pred": None, "error": err[:160].decode(errors="replace"), "cost": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/root/t2a/gclean"))
    ap.add_argument("--cap", type=float, required=True, help="USD, cumulative across runs (ledger)")
    ap.add_argument("--classes", default=",".join(TARGETS))
    ap.add_argument("--push-path", default="yt_windows/gemini_judged.jsonl",
                    help="Hub path for results; give each run its own so one never overwrites another's paid output")
    ap.add_argument("--windows", type=Path, default=None, help="local windows.jsonl (default: the Hub's yt_windows/windows.jsonl)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--push-every", type=int, default=2000, help="upload results to HF every N judged")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, hf_hub_download
    targets = set(a.classes.split(","))
    a.out.mkdir(parents=True, exist_ok=True)
    res_path = a.out / "gemini_judged.jsonl"
    preflight(res_path, a.out / "ledger.json", note="gemini clean")
    ledger = Ledger(a.out / "ledger.json", a.cap)
    sink = JsonlSink(res_path, key="uid")
    done = {u for u in sink.seen}
    key = os.environ["OPENROUTER_API_KEY_GS"]

    by_src = defaultdict(list)
    with open(a.windows or hf_hub_download("aoxo/clap-ft-data", "yt_windows/windows.jsonl", repo_type="dataset"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("label") in targets and r["uid"] not in done:
                by_src[(r["repo"], r["source"])].append({k: r.get(k) for k in ("uid", "label", "source", "start", "dur", "split", "kind")})
    total = sum(len(v) for v in by_src.values())
    log(f"{total} windows to judge in {len(by_src)} videos ({len(done)} already); ledger ${ledger.spent:.2f} / ${a.cap}")

    work: queue.Queue = queue.Queue(maxsize=64)          # bounded: cutting never runs far ahead of judging
    tmp = a.out / "wav"; tmp.mkdir(exist_ok=True)
    stop = threading.Event()

    def producer():
        for (repo, src), rows in by_src.items():
            if stop.is_set(): break
            vid = src.split(":", 1)[1]; local = None
            try:
                local = hf_hub_download(repo, f"audio/{vid}.flac", repo_type="dataset", local_dir=str(a.out / "dl"))
                for r in rows:
                    if stop.is_set(): break
                    w = tmp / (safe_name(r["uid"]) + ".wav")
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{r['start']:.3f}", "-t", f"{r['dur']:.3f}",
                                    "-i", local, "-ac", "1", "-ar", "16000", str(w)], check=True)
                    work.put((r, w))
            except Exception as e:
                log(f"  {src}: {type(e).__name__} {str(e)[:80]}")
            finally:
                if local:
                    try: os.remove(local)
                    except OSError: pass
        for _ in range(a.concurrency): work.put(None)

    n = [0]; stats = Counter(); lock = threading.Lock()
    api = HfApi()

    def push():
        try:
            api.upload_file(path_or_fileobj=str(res_path), path_in_repo=a.push_path,
                            repo_id="aoxo/clap-ft-data", repo_type="dataset",
                            commit_message=f"gemini chapter filter: {n[0]} judged, ${ledger.spent:.2f}")
        except Exception as e:
            log(f"  push failed: {type(e).__name__} {str(e)[:80]}")

    def consumer():
        while True:
            item = work.get()
            if item is None: return
            r, w = item
            if ledger.over():
                stop.set(); w.unlink(missing_ok=True); continue
            j = judge(w, key)
            w.unlink(missing_ok=True)
            ledger.add(j["cost"])
            row = dict(r, **j, keep=(j["pred"] == r["label"]))
            with lock:
                sink.write(row)                         # paid result on disk before anything else
                n[0] += 1
                stats[(r["label"], "keep" if row["keep"] else ("err" if j["pred"] is None else "drop"))] += 1
                if n[0] % 200 == 0:
                    kept = ", ".join(f"{c} {stats[(c, 'keep')]}/{sum(stats[(c, k)] for k in ('keep', 'drop'))}"
                                     for c in sorted(targets))
                    log(f"  {n[0]}/{total} ${ledger.spent:.2f} | kept {kept} | err {sum(v for k, v in stats.items() if k[1] == 'err')}")
                if n[0] % a.push_every == 0: push()

    threads = [threading.Thread(target=producer, daemon=True)] + \
              [threading.Thread(target=consumer, daemon=True) for _ in range(a.concurrency)]
    for t in threads: t.start()
    for t in threads[1:]: t.join()
    sink.close(); push()
    log(f"DONE judged {n[0]}, ledger ${ledger.spent:.2f} / ${a.cap}" + (" (CAP REACHED)" if ledger.over() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
