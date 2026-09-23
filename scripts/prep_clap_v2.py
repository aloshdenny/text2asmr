#!/usr/bin/env python3
"""Stream sources from aoxo/t2a-mommy, cut labeled clips (CPU procs), log-mel on GPU, write fp16 shards.

Output: <out>/shard_XXXX.f16 (raw fp16, N x 1001 x 64) + <out>/index.jsonl (uid,label,text,split,shard,row).
Resumable per source via <out>/done_sources.txt.  GPU mel replicates ClapFeatureExtractor (htk, repeatpad, dB).
"""
from __future__ import annotations
import argparse, json, os, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
SR = 48_000; MAX_S = SR * 10; FRAMES = 1001; MELS = 64; N_FFT = 1024; HOP = 480
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def decode_url(url, token, start, dur):
    """Pull just the clip's byte range straight from the Hub - a 30-min source costs ~1 MB, not ~28 MB."""
    cmd = ["ffmpeg", "-v", "error", "-threads", "1",
           "-headers", f"Authorization: Bearer {token}\r\n", "-reconnect", "1", "-reconnect_streamed", "1",
           "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", url,
           "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=300).stdout
    return np.frombuffer(out, dtype=np.float32).copy()

def decode(path, start, dur):
    cmd = ["ffmpeg", "-v", "error", "-threads", "1", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
    return np.frombuffer(out, dtype=np.float32).copy()

def repeatpad(w: np.ndarray) -> np.ndarray:
    w = w[:MAX_S]
    if w.size < MAX_S:
        n = MAX_S // w.size
        w = np.tile(w, n); w = np.pad(w, (0, MAX_S - w.size))
    return w.astype(np.float32)

def process_source(args):
    src, rows, repo, tmp, shm, stream = args
    from huggingface_hub import hf_hub_download, hf_hub_url
    token = os.environ["HF_TOKEN"]; local = None; wavs, metas, nfail = [], [], 0
    try:
        if stream:
            url = hf_hub_url(repo, src, repo_type="dataset")
            for r in rows:
                for attempt in range(3):
                    try:
                        w = decode_url(url, token, float(r["start"]), float(r["duration"]))
                        if w.size < SR // 4: break
                        wavs.append(repeatpad(w)); metas.append({k: r[k] for k in ("uid", "label", "text", "split")}); break
                    except Exception:
                        if attempt == 2: nfail += 1
                        else: time.sleep(2 * (attempt + 1))
            path = None
            if wavs:
                path = shm / f"{os.getpid()}_{abs(hash(src))}.npy"; np.save(path, np.stack(wavs))
            return src, str(path) if path else None, metas, nfail, None
        for attempt in range(6):
            try:
                local = hf_hub_download(repo, src, repo_type="dataset", token=token, local_dir=str(tmp / f"p{os.getpid()}")); break
            except Exception:
                if attempt == 5: raise
                time.sleep(min(120, 5 * 2 ** attempt))
        for r in rows:
            try:
                w = decode(local, float(r["start"]), float(r["duration"]))
                if w.size < SR // 4: continue
                wavs.append(repeatpad(w)); metas.append({k: r[k] for k in ("uid", "label", "text", "split")})
            except Exception:
                nfail += 1
        path = None
        if wavs:
            path = shm / f"{os.getpid()}_{abs(hash(src))}.npy"; np.save(path, np.stack(wavs))
        return src, str(path) if path else None, metas, nfail, None
    except Exception as e:
        return src, None, [], len(rows), f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        if local:
            try: os.remove(local)
            except Exception: pass

class GpuMel:
    def __init__(self, dev="cuda"):
        import torch
        from transformers import ClapFeatureExtractor
        fe = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        self.torch = torch; self.dev = dev
        self.fb = torch.from_numpy(np.asarray(fe.mel_filters_slaney, dtype=np.float32).T).to(dev)  # (64, 513)
        self.win = torch.hann_window(N_FFT, periodic=True, device=dev)
    def __call__(self, wavs: np.ndarray) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(wavs).to(self.dev)
            spec = t.stft(x, N_FFT, HOP, N_FFT, self.win, center=True, pad_mode="reflect", return_complex=True)
            power = spec.real ** 2 + spec.imag ** 2                       # (B, 513, T)
            mel = t.einsum("mf,bft->btm", self.fb, power)                  # (B, T, 64)
            db = 10.0 * t.log10(t.clamp(mel, min=1e-10))
            return db[:, :FRAMES, :].to(t.float16).cpu().numpy()

def validate(gm):
    from transformers import ClapFeatureExtractor
    fe = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
    rng = np.random.default_rng(0); worst = 0.0
    for n in (SR * 2, SR * 7, MAX_S):
        w = (rng.standard_normal(n) * 0.05).astype(np.float32)
        ref = fe(w, sampling_rate=SR, return_tensors="np")["input_features"][0]; ref = ref[0] if ref.ndim == 3 else ref
        got = gm(repeatpad(w)[None])[0].astype(np.float32)
        worst = max(worst, float(np.abs(ref - got).max()))
    log(f"gpu-mel validation: max_abs_diff_dB={worst:.4f} (ref range ~[-100,+20])")
    if worst > 0.5: raise SystemExit("GPU mel does not match HF extractor")

class ShardWriter:
    def __init__(self, out: Path, per_shard: int):
        self.out = out; self.per = per_shard
        self.sid = len(sorted(out.glob("shard_*.f16"))); self.f = None; self.n = 0
        self.idx = open(out / "index.jsonl", "a"); self._open()
    def _open(self):
        if self.f: self.f.close()
        self.f = open(self.out / f"shard_{self.sid:04d}.f16", "ab"); self.n = 0
    def write(self, feats: np.ndarray, metas: list[dict]):
        for ft, m in zip(feats, metas):
            if self.n >= self.per: self.sid += 1; self._open()
            self.f.write(ft.tobytes()); m = dict(m); m["shard"] = self.sid; m["row"] = self.n
            self.idx.write(json.dumps(m) + "\n"); self.n += 1
        self.f.flush(); self.idx.flush()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", default="aoxo/t2a-mommy"); ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--per-shard", type=int, default=4000); ap.add_argument("--tmp", type=Path, default=Path("/workspace/tmp_src"))
    ap.add_argument("--shm", type=Path, default=Path("/dev/shm/t2a"))
    ap.add_argument("--stream", action="store_true", help="decode clips over HTTP range requests instead of downloading whole sources")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); a.tmp.mkdir(parents=True, exist_ok=True); a.shm.mkdir(parents=True, exist_ok=True)
    gm = GpuMel(); validate(gm)
    done_path = a.out / "done_sources.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    by_src = defaultdict(list)
    for line in open(a.subset):
        r = json.loads(line); by_src[r["source"]].append(r)
    todo = [s for s in by_src if s not in done]
    log(f"sources total={len(by_src)} done={len(done)} todo={len(todo)} clips_todo={sum(len(by_src[s]) for s in todo)}")
    writer = ShardWriter(a.out, a.per_shard); done_f = open(done_path, "a")
    stats = {"src": 0, "clips": 0, "fail": 0}; t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for src, path, metas, nfail, err in pool.imap_unordered(process_source, [(s, by_src[s], a.repo, a.tmp, a.shm, a.stream) for s in todo], chunksize=1):
            if err: log(f"source fail {src}: {err}")
            if path:
                wavs = np.load(path); os.remove(path)
                for i in range(0, len(wavs), 128):
                    writer.write(gm(wavs[i:i+128]), metas[i:i+128])
                stats["clips"] += len(metas)
            stats["fail"] += nfail; stats["src"] += 1
            done_f.write(src + "\n"); done_f.flush()
            if stats["src"] % 100 == 0:
                el = time.time() - t0; rate = stats["src"] / el
                log(f"sources {stats['src']}/{len(todo)} clips={stats['clips']} fail={stats['fail']} "
                    f"{rate*60:.1f} src/min {stats['clips']/el:.0f} clips/s ETA {((len(todo)-stats['src'])/max(rate,1e-6))/60:.0f} min")
    log(f"PREP_DONE sources={stats['src']} clips={stats['clips']} fail={stats['fail']} shards={writer.sid+1}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
