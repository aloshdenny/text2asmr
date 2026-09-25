#!/usr/bin/env python3
"""Rank an unlabeled candidate pool with CLAP so the paid labeler only sees clips worth labeling.

Dense mining multiplies candidates but not their distribution: 2.5x more clips from a corpus that is 62%
whispering is 2.5x more whispering.  Under a fixed A100 budget the thing that matters is *which* clips get
labeled, so this runs in two cheap stages on the free GPU:

  stage 1 (no audio at all): every source already has labels from earlier passes, so rank sources by their
      observed rare-class density.  Free, and it alone concentrates the queue enormously.
  stage 2 (GPU): stream the top-ranked clips straight from the Hub by byte range, score them with the
      fine-tuned CLAP + its calibrated head, and keep the ones the model thinks are a target class.

Output: labels/label_queue.jsonl in the corpus repo -- same schema the labeler consumes, ordered best-first.

  python3 score_pool_clap.py --repo aoxo/t2a-mommy --ckpt ~/t2a/v61/ckpt/best --top 400000 --keep 150000
"""
from __future__ import annotations
import argparse, json, os, queue, subprocess, threading, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SR = 48_000; MAX_S = SR * 10; FRAMES = 1001; MELS = 64; N_FFT = 1024; HOP = 480
RARE_WEIGHT = {"kissing": 3.0, "moaning": 2.0, "mouth sounds": 1.5, "breathing": 0.5}
POOL_PREFIX = "labels/pending_candidates"   # the dense miner writes numbered parts alongside the first pass


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def decode_file(path, start, dur):
    """Cut one clip out of a local file."""
    cmd = ["ffmpeg", "-v", "error", "-threads", "1", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path),
           "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def duration_of(path) -> float:
    """Clip geometry recomputed from alignments can run past the real audio; skip those instead of
    paying ffmpeg to fail on ~19% of requests."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)], capture_output=True, timeout=60).stdout
        return float(out.strip() or 0.0)
    except Exception:
        return 0.0


def repeatpad(w: np.ndarray) -> np.ndarray:
    w = w[:MAX_S]
    if w.size == 0: raise ValueError("empty clip")
    if w.size < MAX_S:
        w = np.tile(w, MAX_S // w.size); w = np.pad(w, (0, MAX_S - w.size))
    return w.astype(np.float32)


class Scorer:
    """Fine-tuned CLAP audio tower + the trained 5-way head, on GPU."""

    def __init__(self, ckpt: Path, dev="cuda"):
        import torch
        from transformers import ClapModel, ClapFeatureExtractor
        self.torch = torch; self.dev = dev
        meta = json.load(open(ckpt / "vocal_meta.json"))
        self.classes = meta["classes"]
        self.model = ClapModel.from_pretrained(str(ckpt)).to(dev).eval()
        self.head = torch.nn.Linear(self.model.config.projection_dim, len(self.classes) + 1).to(dev)
        self.head.load_state_dict(torch.load(ckpt / "head.pt", map_location=dev)); self.head.eval()
        fe = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        self.fb = torch.from_numpy(np.asarray(fe.mel_filters_slaney, dtype=np.float32).T).to(dev)
        self.win = torch.hann_window(N_FFT, periodic=True, device=dev)
        log(f"scorer ready: classes={self.classes}")

    def __call__(self, wavs: np.ndarray) -> np.ndarray:
        """(B, 480000) float32 -> (B, C+1) probabilities, background last."""
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(wavs).to(self.dev)
            spec = t.stft(x, N_FFT, HOP, N_FFT, self.win, center=True, pad_mode="reflect", return_complex=True)
            power = spec.real ** 2 + spec.imag ** 2
            mel = t.einsum("mf,bft->btm", self.fb, power)
            db = (10.0 * t.log10(t.clamp(mel, min=1e-10)))[:, :FRAMES, :]
            feats = db.unsqueeze(1)                                   # (B, 1, 1001, 64)
            is_longer = t.zeros((feats.shape[0], 1), dtype=t.bool, device=self.dev)
            emb = self.model.get_audio_features(input_features=feats, is_longer=is_longer)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return t.softmax(self.head(emb * 10.0), dim=1).float().cpu().numpy()


def source_priors(repo: str, cache: str) -> tuple[dict[str, float], float]:
    """Per-source rare-class density from the labels we already have, plus the corpus mean as the prior
    for sources that have never been labeled."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    per = defaultdict(Counter)
    for f in api.list_repo_files(repo, repo_type="dataset"):
        if not (f.startswith("labels/qwen3omni") and f.endswith(".jsonl")): continue
        try: p = hf_hub_download(repo, f, repo_type="dataset", cache_dir=cache, force_download=True)
        except Exception as e: log(f"  skip {f} ({type(e).__name__})"); continue
        for line in open(p):
            try: r = json.loads(line)
            except Exception: continue
            uid = r.get("uid", ""); src = r.get("source") or (uid.rsplit(".m4a_", 1)[0] + ".m4a" if ".m4a_" in uid else None)
            if not src: continue
            per[src][r["label"]] += 1; per[src]["__n__"] += 1
        try: os.remove(os.path.realpath(p))
        except OSError: pass
    score = {}
    for src, c in per.items():
        n = max(c["__n__"], 1)
        score[src] = sum(RARE_WEIGHT.get(k, 0.0) * v for k, v in c.items()) / n
    mean = float(np.mean(list(score.values()))) if score else 0.0
    log(f"{repo}: priors for {len(score)} sources, corpus mean {mean:.3f}")
    return score, mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--state", type=Path, default=Path("/home/tinkerspace/t2a/score"))
    ap.add_argument("--top", type=int, default=400000, help="clips to actually score on the GPU (stage 2)")
    ap.add_argument("--keep", type=int, default=150000, help="clips to write into the label queue")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12, help="parallel source fetchers feeding the GPU")
    ap.add_argument("--min-target-p", type=float, default=0.35, help="keep a clip if some target class beats this")
    ap.add_argument("--cache", default=os.environ.get("T2A_CACHE", "hfcache"))
    a = ap.parse_args()
    a.state.mkdir(parents=True, exist_ok=True)
    tag = a.repo.split("/")[-1]
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); token = os.environ["HF_TOKEN"]

    labeled = set()
    for f in api.list_repo_files(a.repo, repo_type="dataset"):
        if f.startswith("labels/qwen3omni") and f.endswith(".jsonl"):
            try: p = hf_hub_download(a.repo, f, repo_type="dataset", cache_dir=a.cache, force_download=True)
            except Exception: continue
            for line in open(p):
                try: labeled.add(json.loads(line)["uid"])
                except Exception: pass
            try: os.remove(os.path.realpath(p))
            except OSError: pass
    log(f"{len(labeled)} uids already labeled")

    pool = []
    pool_files = sorted(f for f in api.list_repo_files(a.repo, repo_type="dataset") if f.startswith(POOL_PREFIX))
    log(f"candidate pools: {pool_files}")
    for name in pool_files:
        try: p = hf_hub_download(a.repo, name, repo_type="dataset", cache_dir=a.cache, force_download=True)
        except Exception as e: log(f"no {name} ({type(e).__name__})"); continue
        n = 0
        for line in open(p):
            try: r = json.loads(line)
            except Exception: continue
            if r["uid"] in labeled: continue
            pool.append(r); n += 1
        log(f"{name}: {n} unlabeled candidates")
    if not pool: log("nothing to score"); return 0

    prior, mean = source_priors(a.repo, a.cache)
    pool.sort(key=lambda r: -prior.get(r["source"], mean))
    stage1 = pool[: a.top]
    log(f"stage 1: {len(pool)} candidates -> top {len(stage1)} by source prior "
        f"(best {prior.get(stage1[0]['source'], mean):.3f}, worst kept {prior.get(stage1[-1]['source'], mean):.3f})")

    scorer = Scorer(a.ckpt)
    out_path = a.state / f"scores_{tag}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try: done.add(json.loads(line)["uid"])
            except Exception: pass
        log(f"resuming: {len(done)} already scored")
    todo = [r for r in stage1 if r["uid"] not in done]

    by_source: dict[str, list] = defaultdict(list)
    for r in todo: by_source[r["source"]].append(r)
    sources = list(by_source)
    log(f"{len(todo)} clips over {len(sources)} sources ({len(todo)/max(len(sources),1):.0f} per source)")

    q: queue.Queue = queue.Queue(maxsize=a.batch * 8)
    def producer(srcs):
        for src in srcs:
            local = None
            try:
                local = hf_hub_download(a.repo, src, repo_type="dataset", cache_dir=str(a.state / "audio"))
                dur = duration_of(local)
                for r in by_source[src]:
                    cs, cd = r.get("cut_start", r["start"]), r.get("cut_duration", 8.0)
                    if dur and cs >= dur - 0.2: q.put((r, None)); continue
                    try: q.put((r, repeatpad(decode_file(local, cs, min(cd, max(dur - cs, 0.5)))))) 
                    except Exception: q.put((r, None))
            except Exception:
                for r in by_source[src]: q.put((r, None))
            finally:
                if local:
                    try: os.remove(os.path.realpath(local))
                    except Exception: pass
        q.put(None)

    chunks = [sources[i::a.workers] for i in range(a.workers)]
    threads = [threading.Thread(target=producer, args=(c,), daemon=True) for c in chunks]
    for t in threads: t.start()

    fh = out_path.open("a"); alive = len(threads); n_ok = n_fail = 0; t0 = time.time()
    buf_rows, buf_wavs = [], []
    def flush():
        nonlocal n_ok
        if not buf_rows: return
        probs = scorer(np.stack(buf_wavs))
        for r, p in zip(buf_rows, probs):
            best_i = int(np.argmax(p[:-1])); best_p = float(p[best_i])
            fh.write(json.dumps({"uid": r["uid"], "source": r["source"], "repo": r["repo"],
                                 "start": r["start"], "duration": r["duration"],
                                 "cut_start": r.get("cut_start"), "cut_duration": r.get("cut_duration"),
                                 "clap_class": scorer.classes[best_i], "clap_p": round(best_p, 4),
                                 "clap_bg": round(float(p[-1]), 4)}) + "\n")
        n_ok += len(buf_rows); buf_rows.clear(); buf_wavs.clear()

    while alive:
        item = q.get()
        if item is None: alive -= 1; continue
        r, w = item
        if w is None: n_fail += 1; continue
        buf_rows.append(r); buf_wavs.append(w)
        if len(buf_rows) >= a.batch:
            flush()
            if n_ok % (a.batch * 20) == 0:
                el = time.time() - t0
                log(f"  scored {n_ok}/{len(todo)} ({n_ok/max(el,1e-9):.0f} clips/s, {n_fail} failed) "
                    f"ETA {((len(todo)-n_ok)/max(n_ok/max(el,1e-9),1e-9))/60:.0f} min")
                fh.flush()
    flush(); fh.close()
    log(f"stage 2 done: {n_ok} scored, {n_fail} failed")

    # queue the most promising clips, best-first, capped per class so one class cannot eat the budget
    rows = [json.loads(l) for l in out_path.open()]
    keep = [r for r in rows if r["clap_p"] >= a.min_target_p]
    keep.sort(key=lambda r: -r["clap_p"])
    per_class_cap = max(1, a.keep // max(len(scorer.classes), 1))
    taken: Counter = Counter(); queued = []
    for r in keep:
        if taken[r["clap_class"]] >= per_class_cap: continue
        taken[r["clap_class"]] += 1; queued.append(r)
        if len(queued) >= a.keep: break
    qpath = a.state / f"label_queue_{tag}.jsonl"
    qpath.write_text("".join(json.dumps(r) + "\n" for r in queued))
    log(f"queue: {len(queued)} clips ({dict(taken)}) -> {qpath}")
    api.upload_file(path_or_fileobj=str(qpath), path_in_repo="labels/label_queue.jsonl", repo_id=a.repo,
                    repo_type="dataset", commit_message=f"CLAP-ranked label queue: {len(queued)} clips {dict(taken)}")
    api.upload_file(path_or_fileobj=str(out_path), path_in_repo="labels/clap_pool_scores.jsonl", repo_id=a.repo,
                    repo_type="dataset", commit_message=f"CLAP pool scores: {len(rows)} clips")
    log("SCORE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
