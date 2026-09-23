#!/usr/bin/env python3
"""Label aoxo/t2a-mommy non-speech gaps with CLAP into CLAP-training format.

Target formats (both written):

1. Our corpus metadata (same fields as scripts/build_datasets.py triggers):
   file_name, trigger, intensity, caption, clap_confidence, source, start, ...

2. LAION-CLAP fine-tune sidecar per clip (audio-dataset webdataset schema):
   48 kHz FLAC + sibling .json with {"text": [caption, ...probes], "tag": [key]}

Only silence gaps after the first spoken word that clear the silence floor AND
beat CLAP's negative probes by MARGIN are kept -- that is the non-speech set,
not every ASR gap.

Resumable via Hub: skips sources that already have a committed
"<source>.m4a.clap.json" marker listing accepted uids for that file.

    python3 scripts/clap_label_audios2.py --num-shards 12 --shard-index 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import zlib
from pathlib import Path

import numpy as np

REPO = "aoxo/t2a-mommy"
OUT_REPO = "aoxo/audios2-clap"  # labeled clips + metadata for CLAP fine-tune
CLAP_ID = "laion/clap-htsat-fused"
CLAP_SR = 48_000
MAX_TAG_S = 10.0
MAX_TAG_N = int(MAX_TAG_S * CLAP_SR)
SILENCE_FLOOR_DB = -50.0
BASE = Path(os.environ.get("CLAP_LABEL_BASE", "/workspace/clap_audios2"))
STAGE = BASE / "stage"
OUT = BASE / "out"
LOG = BASE / "clap_label.log"

_log_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(line + "\n")


def hf_token() -> str:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not set")
    return token


def rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-9))


def decode(path: Path, sr: int, mono: bool, start: float, dur: float) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr), "-ac", "1" if mono else "2", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(out, dtype=np.float32)
    if mono:
        return audio.reshape(1, -1)
    return audio.reshape(-1, 2).T if audio.size else audio.reshape(2, 0)


def decode_file_mono(path: Path, sr: int) -> np.ndarray:
    """One ffmpeg for the whole file — feeds the GPU instead of one spawn per gap."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr), "-ac", "1", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def fit_tag_clip(mono: np.ndarray) -> np.ndarray:
    """Fixed 10s windows so every GPU batch has the same shape and stays busy."""
    if mono.shape[0] >= MAX_TAG_N:
        return np.ascontiguousarray(mono[:MAX_TAG_N])
    out = np.zeros(MAX_TAG_N, dtype=np.float32)
    out[: mono.shape[0]] = mono
    return out


def _features(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out


def _htk_mel_filters(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz_to_mel(h: float) -> float:
        return 2595.0 * np.log10(1.0 + h / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(np.int64)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = int(bins[i - 1]), int(bins[i]), int(bins[i + 1])
        for j in range(left, center):
            fbank[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            fbank[i - 1, j] = (right - j) / max(right - center, 1)
    return fbank


class Tagger:
    def __init__(self, device: str, batch: int, clap_id: str = CLAP_ID) -> None:
        import torch
        from transformers import ClapModel, ClapProcessor
        from text2asmr.data.ontology import all_probes

        self.torch = torch
        self.device = device
        self.batch = batch
        self.clap_id = clap_id
        self.model = ClapModel.from_pretrained(clap_id).to(device).eval()
        proc_id = os.environ.get("CLAP_PROCESSOR") or clap_id
        self.processor = ClapProcessor.from_pretrained(proc_id)
        texts, self.owners = all_probes()
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            self.text_emb = _features(
                self.model.get_text_features(**{k: v.to(device) for k, v in inputs.items()})
            )
        self.pos = [i for i, o in enumerate(self.owners) if o is not None]
        self.neg = [i for i, o in enumerate(self.owners) if o is None]
        from text2asmr.data.ontology import BY_KEY, MARGIN
        self.by_key = BY_KEY
        self.margin = MARGIN
        self.mel_fb = None
        self.hann = None
        n_fft = 1024
        fb = _htk_mel_filters(CLAP_SR, n_fft, 64, 50.0, 14000.0)
        self.mel_fb = torch.from_numpy(fb).to(device)
        self.hann = torch.hann_window(n_fft, device=device)
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            dummy = np.zeros((MAX_TAG_N,), dtype=np.float32)
            self.forward_wave([dummy] * min(8, batch))

    def prepare(self, clips: list[np.ndarray]) -> dict:
        try:
            inputs = self.processor(
                audios=clips, sampling_rate=CLAP_SR, return_tensors="pt", padding=True
            )
        except (TypeError, ValueError):
            inputs = self.processor(
                audio=clips, sampling_rate=CLAP_SR, return_tensors="pt", padding=True
            )
        out = {}
        for key, val in inputs.items():
            if self.device == "cuda":
                val = val.pin_memory()
            out[key] = val
        return out

    def forward(self, inputs: dict) -> list[tuple[str, float] | None]:
        torch = self.torch
        use_cuda = self.device == "cuda"
        with torch.inference_mode():
            tensors = {
                k: v.to(self.device, non_blocking=use_cuda) for k, v in inputs.items()
            }
            emb = _features(self.model.get_audio_features(**tensors))
            scores = emb @ self.text_emb.T
            results: list[tuple[str, float] | None] = []
            pos = scores[:, self.pos]
            neg = scores[:, self.neg]
            best_pos, pos_idx = pos.max(dim=1)
            best_neg, _ = neg.max(dim=1)
            keep = (best_pos - best_neg) >= self.margin
            for i in range(scores.shape[0]):
                if not bool(keep[i]):
                    results.append(None)
                else:
                    key = self.owners[self.pos[int(pos_idx[i])]]
                    results.append((key, float(best_pos[i])))
            return results

    def forward_wave(self, clips: list[np.ndarray] | None = None, wav: "object | None" = None) -> list[tuple[str, float] | None]:
        """STFT + log-mel + HTSAT all on GPU so the card actually stays busy."""
        torch = self.torch
        if wav is None:
            wav = torch.from_numpy(np.stack(clips, axis=0))
            if self.device == "cuda":
                wav = wav.pin_memory().to(self.device, non_blocking=True)
            else:
                wav = wav.to(self.device)
        elif self.device == "cuda" and not wav.is_cuda:
            wav = wav.to(self.device, non_blocking=True)
        n_fft = 1024
        target_frames = 1001
        with torch.inference_mode():
            spec = torch.stft(
                wav, n_fft=n_fft, hop_length=480, win_length=n_fft,
                window=self.hann, center=True, return_complex=True,
            )
            power = spec.real.square() + spec.imag.square()
            # torch.stft is (B, freq, time); keep it that way.
            if power.shape[1] != n_fft // 2 + 1 and power.shape[-1] == n_fft // 2 + 1:
                power = power.transpose(1, 2)
            mel = torch.einsum("mf,bft->bmt", self.mel_fb, power)
            if mel.shape[-1] < target_frames:
                mel = torch.nn.functional.pad(mel, (0, target_frames - mel.shape[-1]))
            else:
                mel = mel[..., :target_frames]
            log_mel = torch.log(mel.clamp(min=1e-6))
            # Processor layout is (B, 1, time=1001, n_mels=64), not (B, 1, 64, time).
            feats = log_mel.transpose(1, 2).unsqueeze(1)
            is_longer = self.torch.zeros((feats.shape[0], 1), dtype=self.torch.bool, device=self.device)
            emb = _features(self.model.get_audio_features(input_features=feats, is_longer=is_longer))
            scores = emb @ self.text_emb.T
            pos = scores[:, self.pos]
            neg = scores[:, self.neg]
            best_pos, pos_idx = pos.max(dim=1)
            best_neg, _ = neg.max(dim=1)
            keep = (best_pos - best_neg) >= self.margin
            results: list[tuple[str, float] | None] = []
            for i in range(scores.shape[0]):
                if not bool(keep[i]):
                    results.append(None)
                else:
                    key = self.owners[self.pos[int(pos_idx[i])]]
                    results.append((key, float(best_pos[i])))
            return results

    def tag(self, clips: list[np.ndarray]) -> list[tuple[str, float] | None]:
        results: list[tuple[str, float] | None] = []
        for i in range(0, len(clips), self.batch):
            chunk = clips[i : i + self.batch]
            results.extend(self.forward(self.prepare(chunk)))
        return results

    def captions_for(self, key: str, intensity: str) -> list[str]:
        """LAION-style text list: primary ASMR caption + ontology probes."""
        primary = f"ASMR {intensity} {key}, close-mic binaural, no speech"
        probes = list(self.by_key[key].probes) if key in self.by_key else []
        # Dedupe while preserving order.
        seen = set()
        out = []
        for t in [primary, *probes, f"The sound of {key}"]:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


def drop_cache(path: Path) -> None:
    try:
        blob = Path(os.path.realpath(path))
    except OSError:
        blob = None
    path.unlink(missing_ok=True)
    if blob is not None and blob != path and blob.exists() and STAGE in blob.parents:
        blob.unlink(missing_ok=True)


def shard_paths(paths: list[str], num_shards: int, shard_index: int) -> list[str]:
    return [
        p for p in paths
        if zlib.crc32(p.encode()) % num_shards == shard_index
    ]


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--download-workers", type=int, default=8)
    ap.add_argument("--limit-sources", type=int, default=0)
    ap.add_argument("--push-every", type=int, default=128,
                    help="Hub commit every N pending ops (flac/json/markers)")
    ap.add_argument("--clap-model", default=os.environ.get("CLAP_MODEL", CLAP_ID),
                    help="HF id or local path for ClapModel weights")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    import soundfile as sf
    from text2asmr.data.ontology import intensity_from_loudness
    from text2asmr.data.segment import load_alignment, split_alignment

    BASE.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    (OUT / "triggers").mkdir(parents=True, exist_ok=True)

    def _retry_rate_limited(fn, desc: str):
        # These startup calls run before the pipeline exists, so nothing
        # here is behind the commit_worker's retry loop -- a 429 (the
        # account-wide 1000-req/5min cap, shared across all 6 shards, is
        # hit constantly per the watchdog's own restart log) used to
        # propagate straight out of main() and kill the process outright.
        # The watchdog then relaunches it, which re-issues this exact same
        # call into an account that's still rate limited, crashing again --
        # a tight crash loop (observed directly: "session not running" then
        # "stale" log lines ~60-70s apart, repeated for many minutes) that
        # makes zero progress while adding more requests to an already-hot
        # window, the opposite of what should happen. Retry only on the
        # rate-limit signature specifically; any other exception is
        # re-raised immediately so callers' existing fallback behavior
        # (e.g. "markers/ doesn't exist yet on a fresh repo") is unaffected.
        delay = 10.0
        while True:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                if "429" not in str(exc) and "rate limit" not in str(exc).lower():
                    raise
                log(f"{desc}: rate limited, retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 300.0)

    api = HfApi(token=hf_token())
    try:
        api.create_repo(OUT_REPO, repo_type="dataset", exist_ok=True, private=True)
    except Exception as exc:  # noqa: BLE001
        log(f"create_repo note: {exc}")

    files = _retry_rate_limited(
        lambda: api.list_repo_files(REPO, repo_type="dataset"), "list_repo_files"
    )
    available = set(files)
    sources = sorted(
        p for p in files
        if p.endswith(".m4a") and p + ".json" in available
    )
    # Resume: skip sources that already have a clap marker on OUT_REPO.
    # Scoped to the markers/ subtree only -- OUT_REPO's triggers/ subtree has
    # grown to 100k+ flac+json files, and an unscoped list_repo_files walks
    # the whole repo tree, making startup take many minutes for no reason.
    try:
        marker_entries = _retry_rate_limited(
            lambda: api.list_repo_tree(
                OUT_REPO, path_in_repo="markers", recursive=True, repo_type="dataset",
            ),
            "marker listing",
        )
        done_sources = {
            entry.path[len("markers/"): -len(".clap.json")]
            for entry in marker_entries
            if entry.path.endswith(".clap.json")
        }
    except Exception as exc:  # noqa: BLE001 - markers/ may not exist yet on a fresh repo
        log(f"marker listing note: {exc}")
        done_sources = set()
    sources = [s for s in sources if s not in done_sources]
    sources = shard_paths(sources, args.num_shards, args.shard_index)
    if args.limit_sources:
        sources = sources[: args.limit_sources]
    log(
        f"shard={args.shard_index}/{args.num_shards} todo_sources={len(sources)} "
        f"already_marked={len(done_sources)}"
    )

    device = args.device
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
            log("CUDA unavailable, falling back to CPU")
    tagger = Tagger(device, args.batch, clap_id=args.clap_model)
    log(f"CLAP model={args.clap_model}")
    log(f"CLAP ready on {device}")

    meta_path = OUT / "triggers" / f"metadata_shard{args.shard_index:02d}.jsonl"
    meta_f = meta_path.open("a")
    pending_ops: list[CommitOperationAdd] = []
    # Backpressure on local disk usage, not just commit frequency: a single
    # Hub rate-limit stall (observed to last ~1h) lets accepted clips pile up
    # on RunPod's small ~20GB volume faster than any push_every tuning can
    # safely absorb -- GPU throughput measured at ~110k accepted clips/hour,
    # far beyond what an hour-long stall's worth of local files can hold.
    # Pausing new writes once unflushed local bytes cross HIGH_WATER (and
    # resuming once commits drain it back under LOW_WATER) keeps disk usage
    # bounded regardless of how far behind Hub commits fall.
    pending_bytes = 0
    pending_bytes_lock = threading.Lock()
    HIGH_WATER = 4 * 1024**3
    LOW_WATER = 2 * 1024**3
    # RunPod's /workspace is a network filesystem (FUSE-mounted MooseFS),
    # not local disk -- mkdir(exist_ok=True) is a no-op after the first
    # call for a given hash-prefix dir, but the call itself is still a
    # network round-trip every time on this mount. With writer threads now
    # parallel and thousands of clips flowing through, that was enough
    # accumulated latency to reproduce the exact "batches ms fast, gaps
    # growing" symptom already fixed twice this session -- one call per new
    # directory, not one per clip, is the actual fix.
    known_dirs: set[Path] = set()
    known_dirs_lock = threading.Lock()
    accepted = 0
    rejected = 0
    silent = 0
    t0 = time.time()
    # Hub uploads are slow vs GPU tagging; do them off the inference thread
    # so util can rise and the ramp can add the next card. maxsize=4 here
    # used to mean the GPU/decode/accept pipeline stalled completely the
    # moment the Hub's 128-commits/hour cap hit and one item got stuck
    # retrying for up to an hour -- every op just holds a *path* to a file
    # already durably written to local disk, so queuing many pending
    # batches costs almost nothing and lets production keep running fully
    # decoupled from Hub availability.
    # Unbounded: write_tagged() (now called from several writer threads)
    # calls flush_ops() synchronously, which puts onto this queue -- a
    # maxsize here reproduces the exact same invisible-blocking bug the
    # write_q fix just solved, just one level deeper, the moment Hub
    # uploads fall behind production (which is now much faster). Items are
    # cheap (local file paths, not bytes), so there's no real memory
    # argument for capping this.
    commit_q: queue.Queue = queue.Queue()

    def commit_worker() -> None:
        nonlocal pending_bytes
        while True:
            item = commit_q.get()
            if item is None:
                commit_q.task_done()
                return
            ops, n_ops, n_bytes = item
            delay = 5.0
            while True:
                try:
                    api.create_commit(
                        repo_id=OUT_REPO,
                        repo_type="dataset",
                        operations=ops,
                        commit_message=(
                            f"clap labels shard={args.shard_index} +{n_ops} ops"
                        ),
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    log(f"commit retry: {exc}")
                    # The Hub's per-repo commit cap is genuinely an hour-long
                    # block, not a transient blip -- hammering it every
                    # ~120s (the old cap) for up to 30 tries just wastes
                    # cycles and can extend how long the window stays hot.
                    # Recognize that specific message and wait long enough
                    # to actually clear it in one sleep.
                    # The Hub's own error text for this one says "retry in
                    # about 1 hour" -- the previous 600s cap meant 6 wasted
                    # retries into the same still-hot window before it
                    # actually cleared, observed directly when 4 shards
                    # sharing OUT_REPO's 128/h commit budget all tripped it
                    # simultaneously and stayed blocked well past 600s*2.
                    if "rate limit" in str(exc).lower() and "commits" in str(exc).lower():
                        delay = 3700.0
                        time.sleep(delay)
                        continue
                    time.sleep(delay)
                    delay = min(delay * 2, 600.0)
            for op in ops:
                local = getattr(op, "path_or_fileobj", None)
                if isinstance(local, str):
                    Path(local).unlink(missing_ok=True)
            with pending_bytes_lock:
                pending_bytes -= n_bytes
            log(f"commit ok ops={n_ops}")
            commit_q.task_done()

    commit_thread = threading.Thread(target=commit_worker, daemon=True)
    commit_thread.start()

    ops_lock = threading.Lock()

    def flush_ops(force: bool = False) -> None:
        nonlocal pending_ops, pending_bytes
        with ops_lock:
            if not pending_ops:
                return
            if not force and len(pending_ops) < args.push_every:
                return
            n_ops = len(pending_ops)
            batch_ops = list(pending_ops)
            pending_ops = []
        n_bytes = 0
        for op in batch_ops:
            local = getattr(op, "path_or_fileobj", None)
            if isinstance(local, str):
                try:
                    n_bytes += Path(local).stat().st_size
                except OSError:
                    pass
        with pending_bytes_lock:
            pending_bytes += n_bytes
        commit_q.put((batch_ops, n_ops, n_bytes))
        log(f"commit queued ops={n_ops} depth={commit_q.qsize()} pending_bytes={pending_bytes/1e9:.2f}GB")

    def wait_for_disk_headroom() -> None:
        # Called from writer threads (downstream of the bounded write_q) so
        # blocking here naturally backpressures the whole pipeline, GPU loop
        # included, instead of writing an unbounded amount to disk while Hub
        # commits fall behind.
        with pending_bytes_lock:
            over = pending_bytes > HIGH_WATER
        if not over:
            return
        log(f"pausing writes: pending_bytes={pending_bytes/1e9:.2f}GB > high water")
        while True:
            time.sleep(5.0)
            with pending_bytes_lock:
                if pending_bytes < LOW_WATER:
                    break
        log(f"resuming writes: pending_bytes={pending_bytes/1e9:.2f}GB < low water")

    stats_lock = threading.Lock()
    src_lock = threading.Lock()
    src_state: dict[str, dict] = {}
    processed = 0

    def bump(silent_n: int = 0, rejected_n: int = 0, accepted_n: int = 0) -> None:
        nonlocal silent, rejected, accepted
        with stats_lock:
            silent += silent_n
            rejected += rejected_n
            accepted += accepted_n

    def finish_source(src: str) -> None:
        nonlocal processed, pending_ops
        with src_lock:
            st = src_state.get(src)
            if st is None or st["done"]:
                return
            if st["pending"] > 0:
                return
            st["done"] = True
        file_accepted = st["rows"]
        marker = {
            "source": src,
            "accepted": len(file_accepted),
            "candidates": st["n_cands"],
            "uids": [r["uid"] for r in file_accepted],
        }
        marker_path = OUT / "markers" / (src + ".clap.json")
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker))
        with ops_lock:
            pending_ops.append(CommitOperationAdd(
                path_in_repo=f"markers/{src}.clap.json",
                path_or_fileobj=str(marker_path),
            ))
        flush_ops()
        drop_cache(st["ap"])
        drop_cache(st["jp"])
        with stats_lock:
            processed += 1
            n_proc = processed
            acc, rej, sil = accepted, rejected, silent
        if n_proc % 5 == 0 or n_proc == len(sources):
            rate = n_proc / max(time.time() - t0, 1e-6) * 3600
            log(
                f"[{n_proc}/{len(sources)}] accepted={acc} rejected={rej} "
                f"silent={sil} {rate:.0f} sources/h"
            )

    def write_tagged(meta: dict, tagged: tuple[str, float] | None) -> None:
        src = meta["src"]
        try:
            if tagged is None:
                bump(rejected_n=1)
                return
            wait_for_disk_headroom()
            key, conf = tagged
            try:
                stereo = decode(meta["ap"], CLAP_SR, False, meta["span"].start, meta["span"].duration)
            except subprocess.CalledProcessError:
                stereo = None
            if stereo is None or stereo.size == 0:
                bump(rejected_n=1)
                return
            intensity, scaled = intensity_from_loudness(meta["level"])
            safe_uid = meta["span"].uid.replace("/", "__")
            # Some source titles are long enough that "<safe_uid>.flac"
            # exceeds the 255-byte filename limit most filesystems enforce
            # (this MooseFS mount included). Past that limit sf.write()
            # raises a generic LibsndfileError ("System error", really
            # ENAMETOOLONG) for *every* clip of that source, on *every*
            # pass forever, since the marker never gets written either -
            # pure wasted decode/write work with zero chance of success.
            # Truncate and disambiguate with a hash so long-titled sources
            # still write successfully instead of failing silently forever.
            if len(safe_uid.encode()) + len(".flac") > 255:
                digest = hashlib.md5(safe_uid.encode()).hexdigest()[:10]
                budget = 255 - len(".flac") - len(digest) - 1
                safe_uid = safe_uid.encode()[:budget].decode("utf-8", "ignore") + "_" + digest
            # Flat /triggers/ hit the Hub's 10k-files-per-directory cap last
            # run. Hash-shard so no single directory ever approaches that.
            uid_hash = hashlib.md5(safe_uid.encode()).hexdigest()
            hash_prefix = f"s{args.shard_index:02d}/{uid_hash[:2]}/{uid_hash[2:4]}"
            rel = f"{hash_prefix}/{safe_uid}.flac"
            flac_path = OUT / "triggers" / rel
            parent = flac_path.parent
            if parent not in known_dirs:
                parent.mkdir(parents=True, exist_ok=True)
                with known_dirs_lock:
                    known_dirs.add(parent)
            sf.write(flac_path, stereo.T, CLAP_SR)
            captions = tagger.captions_for(key, intensity)
            row = {
                "file_name": rel,
                "trigger": key,
                "intensity": intensity,
                "intensity_scaled": round(scaled, 3),
                "tag": f"[{intensity}][{key}]",
                "caption": captions[0],
                "text": captions,
                "clap_confidence": round(conf, 4),
                "rms_db": round(meta["level"], 2),
                "source": src,
                "start": round(meta["span"].start, 3),
                "duration": round(meta["span"].duration, 3),
                "channels": int(stereo.shape[0]),
                "uid": meta["span"].uid,
            }
            meta_f.write(json.dumps(row) + "\n")
            meta_f.flush()
            side_path = flac_path.with_suffix(".json")
            side_path.write_text(json.dumps({
                "text": captions,
                "tag": [key],
                "original_data": {
                    "source": src,
                    "start": row["start"],
                    "duration": row["duration"],
                    "intensity": intensity,
                    "clap_confidence": row["clap_confidence"],
                    "uid": meta["span"].uid,
                },
            }))
            with ops_lock:
                pending_ops.append(CommitOperationAdd(
                    path_in_repo=f"triggers/{rel}", path_or_fileobj=str(flac_path),
                ))
                pending_ops.append(CommitOperationAdd(
                    path_in_repo=f"triggers/{rel}.json", path_or_fileobj=str(side_path),
                ))
            with src_lock:
                src_state[src]["rows"].append(row)
            bump(accepted_n=1)
            flush_ops()
        finally:
            with src_lock:
                src_state[src]["pending"] -= 1
            finish_source(src)

    download_q: queue.Queue = queue.Queue(maxsize=args.download_workers * 2)
    work_q: queue.Queue = queue.Queue()
    for s in sources:
        work_q.put(s)
    for _ in range(args.download_workers):
        work_q.put(None)

    def producer() -> None:
        from huggingface_hub import hf_hub_download as dl
        while True:
            src = work_q.get()
            if src is None:
                download_q.put(None)
                return
            try:
                jp = Path(dl(REPO, src + ".json", repo_type="dataset",
                             local_dir=STAGE, token=hf_token()))
                ap_ = Path(dl(REPO, src, repo_type="dataset",
                              local_dir=STAGE, token=hf_token()))
                download_q.put((src, jp, ap_))
            except Exception as exc:  # noqa: BLE001
                log(f"download fail {src}: {exc}")

    for _ in range(args.download_workers):
        threading.Thread(target=producer, daemon=True).start()

    # Keep several full GPU batches ready so inference never waits on ffmpeg.
    clip_q: queue.Queue = queue.Queue(maxsize=max(args.batch * 8, 512))
    write_q: queue.Queue = queue.Queue(maxsize=max(args.batch * 4, 256))
    # Decode releases the GIL during actual ffmpeg subprocess waits, but the
    # surrounding Python bookkeeping (numpy ops, queue puts, thread
    # scheduling) still contends for it -- scaling this all the way to host
    # core count (tried: 116 threads on a 120-core box) made per-batch GPU
    # time balloon from ~150ms to multiple *seconds*, the opposite of the
    # goal, by starving the GPU-issuing thread of scheduling time. This is a
    # moderate bump from the old fixed floor of 8, not a scale-to-cores fix.
    n_decode = min(max(args.download_workers, 8), 32)

    def decode_worker() -> None:
        while True:
            item = download_q.get()
            if item is None:
                clip_q.put(None)
                return
            src, jp, ap_ = item
            try:
                entries = load_alignment(jp)
                cands = [s for s in split_alignment(entries, src) if s.kind == "trigger_candidate"]
                file_end = max((float(e["end"]) for e in entries if "end" in e), default=0.0)
                try:
                    if file_end > 40 * 60:
                        full = None
                    else:
                        full = decode_file_mono(ap_, CLAP_SR)
                except subprocess.CalledProcessError as exc:
                    log(f"decode fail {src}: {exc}")
                    continue
                to_put: list[dict] = []
                silent_n = 0
                for span in cands:
                    if full is not None:
                        i0 = max(int(span.start * CLAP_SR), 0)
                        i1 = min(int(span.end * CLAP_SR), int(full.shape[0]))
                        if i1 <= i0:
                            continue
                        mono = full[i0:i1]
                    else:
                        try:
                            wav = decode(ap_, CLAP_SR, True, span.start, span.duration)
                        except subprocess.CalledProcessError:
                            continue
                        mono = wav[0] if wav.size else wav
                    if mono.size == 0:
                        continue
                    level = rms_db(mono.reshape(1, -1))
                    if level < SILENCE_FLOOR_DB:
                        silent_n += 1
                        continue
                    to_put.append({
                        "src": src, "span": span, "level": level,
                        "mono": fit_tag_clip(np.ascontiguousarray(mono)),
                        "ap": ap_,
                    })
                bump(silent_n=silent_n)
                with src_lock:
                    src_state[src] = {
                        "ap": ap_, "jp": jp, "n_cands": len(cands),
                        "pending": len(to_put), "rows": [], "done": False,
                    }
                for item in to_put:
                    clip_q.put(item)
                if not to_put:
                    finish_source(src)
            except Exception as exc:  # noqa: BLE001
                log(f"process fail {src}: {type(exc).__name__}: {exc}")

    for _ in range(n_decode):
        threading.Thread(target=decode_worker, daemon=True).start()

    def writer() -> None:
        while True:
            item = write_q.get()
            if item is None:
                return
            metas, tags = item
            for meta, tagged in zip(metas, tags):
                try:
                    write_tagged(meta, tagged)
                except Exception as exc:  # noqa: BLE001
                    log(f"write fail {meta.get('src')}: {type(exc).__name__}: {exc}")

    # A single writer thread doing synchronous FLAC encode + disk I/O per
    # clip couldn't keep up once the decode fix actually filled clip_q --
    # write_q.put() in the main GPU loop below then blocked for tens of
    # seconds per call, invisibly (it's timed *after* the dt measurement),
    # which is exactly what made batches 70-90s apart despite each one still
    # reporting ~60ms of real GPU time. A modest handful of writer threads
    # (not scaled to core count -- that overcorrection already burned us
    # once on the decode side) spreads the encode/IO work out.
    n_write = 4
    write_threads = [threading.Thread(target=writer, daemon=True) for _ in range(n_write)]
    for t in write_threads:
        t.start()

    decode_finished = 0
    decode_lock = threading.Lock()
    decode_done = threading.Event()

    def take_batch() -> tuple[list[np.ndarray], list[dict]] | None:
        nonlocal decode_finished
        clips: list[np.ndarray] = []
        metas: list[dict] = []
        while len(clips) < args.batch:
            try:
                item = clip_q.get(timeout=0.2)
            except queue.Empty:
                if clips and decode_done.is_set():
                    break
                if decode_done.is_set() and clip_q.empty() and not clips:
                    return None
                continue
            if item is None:
                with decode_lock:
                    decode_finished += 1
                    if decode_finished >= n_decode:
                        decode_done.set()
                continue
            clips.append(item["mono"])
            metas.append(item)
        return (clips, metas) if clips else None

    stacked_q: queue.Queue = queue.Queue(maxsize=6)

    def stack_worker() -> None:
        import torch
        while True:
            batch = take_batch()
            if batch is None:
                stacked_q.put(None)
                return
            clips, metas = batch
            wav = torch.from_numpy(np.stack(clips, axis=0)).pin_memory()
            stacked_q.put((wav, metas))

    threading.Thread(target=stack_worker, daemon=True).start()

    gpu_batches = 0
    gpu_t0 = time.time()
    while True:
        item = stacked_q.get()
        if item is None:
            break
        wav, metas = item
        t_gpu = time.time()
        tags = tagger.forward_wave(wav=wav)
        gpu_batches += 1
        dt = time.time() - t_gpu
        if gpu_batches <= 5 or gpu_batches % 10 == 0:
            log(
                f"gpu batch={gpu_batches} n={len(metas)} {dt*1000:.0f}ms "
                f"clip_q={clip_q.qsize()} stacked_q={stacked_q.qsize()} "
                f"{gpu_batches / max(time.time() - gpu_t0, 1e-6) * 3600:.0f} batches/h"
            )
        write_q.put((metas, tags))

    for _ in write_threads:
        write_q.put(None)
    for t in write_threads:
        t.join(timeout=600)

    flush_ops(force=True)
    commit_q.put(None)
    commit_q.join()
    commit_thread.join(timeout=600)
    # Push shard metadata jsonl
    if meta_path.exists() and meta_path.stat().st_size:
        api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo=f"triggers/{meta_path.name}",
            repo_id=OUT_REPO,
            repo_type="dataset",
        )
    log(
        f"CLAP_DONE shard={args.shard_index} accepted={accepted} rejected={rejected} "
        f"silent={silent} -> {OUT_REPO}"
    )
    meta_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
