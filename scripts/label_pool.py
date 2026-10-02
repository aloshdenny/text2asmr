#!/usr/bin/env python3
"""Large-scale weak supervision, step 1: label a stratified slice of the CLAP pool with every labeller.

Weak supervision averages out label noise by asking many imperfect labellers and weighting each by how often it
agrees with a human on each class (fuse_labels.py does that part). This job collects the votes:
  * audio LLMs via OpenRouter, given the same 19-label menu the human kits use (Gemini 3.1 Pro on the org's Google
    key, MiMo v2.6 flash on OpenRouter credit)
  * AST (AudioSet) mapped onto our classes, and a loudness rule for silence -- free, run locally
Clips are the windows the human kits use (from 0.5 s before the clip start, 6 s long), cut straight from the
Hub copy of the recording (ffmpeg seeks over HTTP); clips are kept for the local labellers (AST, loudness).

Pool: v7 stage-2 index (aoxo/clap-ft-data v7s2/index.jsonl), train split, --per-class clips per pipeline label,
at most --per-recording from one recording, in-domain clips mapped to their corpus repo, YT clips from
asmr-yt-chapters. A dollar ledger caps the LLM spend.

  python label_pool.py --out D:\\t2a\\pool --per-class 1700 --models gemini-3.1-pro,mimo-v2.6-flash --cap 220
"""
from __future__ import annotations
import argparse, base64, json, os, random, re, subprocess, sys, tempfile, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_agreement import LABELS, MODELS, ask

CORPORA = ("aoxo/t2a-mommy", "aoxo/t2a-daddy")
YT_REPO = "aoxo/asmr-yt-chapters"


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


class Ledger:
    def __init__(self, path: Path, cap: float):
        self.path, self.cap, self.lock = path, cap, threading.Lock()
        self.spent = json.loads(path.read_text())["spent"] if path.exists() else 0.0

    def add(self, x: float):
        with self.lock:
            self.spent += x; self.path.write_text(json.dumps({"spent": self.spent, "cap": self.cap}))

    def over(self) -> bool: return self.spent >= self.cap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=1700)
    ap.add_argument("--per-recording", type=int, default=4)
    ap.add_argument("--models", default="gemini-3.1-pro,mimo-v2.6-flash")
    ap.add_argument("--cut-only", action="store_true", help="only cut the clips (no LLM labels, no spend), e.g. to queue them on the crowd site")
    ap.add_argument("--cap", type=float, default=220.0, help="USD cap across all LLM labellers in this job")
    ap.add_argument("--dl-workers", type=int, default=24, help="clips fetched and labelled at once")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-file", type=Path, default=None, help="label these clips (jsonl uid/label/src/repo/rec/start) instead of sampling")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download
    a.out.mkdir(parents=True, exist_ok=True)
    key = None if a.cut_only else os.environ["OPENROUTER_API_KEY_GS"]
    ledger = Ledger(a.out / "ledger.json", a.cap)

    # ---- the stratified pool ----
    pool_f = a.pool_file or (a.out / "pool.jsonl")
    if not pool_f.exists():
        idx = [json.loads(l) for l in open(hf_hub_download("aoxo/clap-ft-data", "v7s2/index.jsonl", repo_type="dataset",
                                                             local_dir=str(a.out / "dl")), encoding="utf-8")]
        owner = {}
        for repo in CORPORA:
            for f in ("labels/qwen3omni.jsonl", "labels/qwen3omni_expansion.jsonl"):
                try: p = hf_hub_download(repo, f, repo_type="dataset", local_dir=str(a.out / "dl" / repo.split("/")[1]))
                except Exception: continue
                for l in open(p, encoding="utf-8"):
                    try: owner.setdefault(json.loads(l)["uid"].rsplit("_", 1)[0], repo)
                    except Exception: pass
        rng = random.Random(a.seed); by = defaultdict(list)
        for r in idx:
            if r["split"] != "train": continue
            if r["src"] == "indomain":
                rec = r["uid"].rsplit("_", 1)[0]
                if rec not in owner: continue
                r["repo"], r["rec"], r["start"] = owner[rec], rec, int(r["uid"].rsplit("_", 1)[1]) / 1000 - 0.5
            else:                                                       # yt:<id>:<start s>
                _, vid, s = r["uid"].split(":"); r["repo"], r["rec"], r["start"] = YT_REPO, vid, float(s)
            by[r["label"]].append(r)
        picks = []
        for lab, rs in sorted(by.items()):
            rng.shuffle(rs); per_rec = defaultdict(int); n = 0
            for r in rs:
                if n >= a.per_class: break
                if per_rec[r["rec"]] >= a.per_recording: continue
                per_rec[r["rec"]] += 1; n += 1
                picks.append({k: r[k] for k in ("uid", "label", "src", "repo", "rec", "start")})
        pool_f.write_text("".join(json.dumps(p) + "\n" for p in picks), encoding="utf-8")
        log(f"pool: {len(picks)} clips, {len({p['rec'] for p in picks})} recordings")
    pool = [json.loads(l) for l in open(pool_f, encoding="utf-8")]

    # ---- yt audio paths: the chapter repo stores one audio file per video ----
    yt_files = {}
    if any(p["repo"] == YT_REPO for p in pool):
        from huggingface_hub import HfApi
        for f in HfApi().list_repo_files(YT_REPO, repo_type="dataset"):
            m = re.match(r"audio/([A-Za-z0-9_-]{11})\.(m4a|webm|opus|mp3|flac|wav)$", f)
            if m: yt_files.setdefault(m.group(1), f)

    models = [] if a.cut_only else [m for m in a.models.split(",") if m]
    sinks = {m: a.out / f"{m}.jsonl" for m in models}
    done = {m: ({json.loads(l)["clip"] for l in open(f, encoding="utf-8") if not json.loads(l)["raw"].startswith("ERROR")}
                if f.exists() else set()) for m, f in sinks.items()}
    lock = threading.Lock()
    by_rec = defaultdict(list)
    for p in pool:
        if a.cut_only or any(p["uid"] not in done[m] for m in models): by_rec[(p["repo"], p["rec"])].append(p)
    log(f"{sum(len(v) for v in by_rec.values())} clips in {len(by_rec)} recordings still to label; ledger ${ledger.spent:.2f}/{a.cap}")
    tmp = Path(tempfile.mkdtemp(prefix="pool_", dir=str(a.out)))
    n_done = [0]

    def label_clip(p, mp3: bytes):
        for m in models:
            if p["uid"] in done[m] or ledger.over(): continue
            labels, raw, cost = ask(MODELS[m][0], mp3, key)
            ledger.add(cost)
            with lock:
                with open(sinks[m], "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"clip": p["uid"], "model": MODELS[m][0], "labels": labels, "raw": raw, "cost": cost, "menu": LABELS}) + "\n")
        with lock:
            n_done[0] += 1
            if n_done[0] % 250 == 0: log(f"  {n_done[0]} clips labelled, ledger ${ledger.spent:.2f}")

    from urllib.parse import quote

    def do_clip(p):
        """Cut the 6 s window straight from the Hub copy: ffmpeg seeks over HTTP and fetches only those seconds
        (whole recordings would be hundreds of GB for 20k clips)."""
        if ledger.over(): return
        path = yt_files.get(p["rec"]) if p["repo"] == YT_REPO else p["rec"]
        if not path: return
        url = f"https://huggingface.co/datasets/{p['repo']}/resolve/main/{quote(path)}"
        out = tmp / "clips" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', p['uid'])}.mp3"; out.parent.mkdir(exist_ok=True)
        for i in range(3):
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, p['start']):.3f}", "-t", "6", "-i", url,
                                "-ac", "1", "-ar", "24000", "-b:a", "48k", str(out)], capture_output=True, timeout=180)
            if r.returncode == 0 and out.exists() and out.stat().st_size > 2000: break
            time.sleep(5 * 2 ** i)
        else: return
        label_clip(p, out.read_bytes())

    todo = [p for ps in by_rec.values() for p in ps]
    with ThreadPoolExecutor(a.dl_workers) as ex: list(ex.map(do_clip, todo))
    log(f"POOL_LLM_DONE ledger ${ledger.spent:.2f}; clips kept in {tmp / 'clips'} for AST/energy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
