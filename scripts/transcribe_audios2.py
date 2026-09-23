#!/usr/bin/env python3
"""Transcribe aoxo/t2a-mommy into word-level alignments, for training v2.

Three concurrent stages, each bottlenecked on a different resource, so none
of them sit idle waiting on the others:

  producer (network)  -> download_q -> transcriber (GPU) -> upload_q -> uploader (network)

Worker count per stage is configurable independently, since the right
balance depends on the host: on tinkerspace's RTX 4000, sharing the card
with trigger training, one lightweight transcriber (~1-2GB VRAM) is the
point -- verified empirically by loading the model standalone and checking
nvidia-smi. On a dedicated high-end GPU (A100/H100 on RunPod), a single
faster-whisper call doesn't come close to saturating the card's FLOPs, so
--transcribe-workers spawns that many independent model instances (each
gets its own WhisperModel; ctranslate2 models aren't documented as safe for
concurrent transcribe() calls from multiple threads on one instance, so
separate instances is the correct way to parallelize, not a workaround) and
--producer-workers/--uploader-workers add threads on the network-bound
stages so they can keep that many transcribers fed.

Alignment JSON matches aoxo/t2a-audios-v1's own schema (a flat list of
{"type": "word"|"silence", "start", "end", ...}), the same shape segment.py
already parses -- so this output slots directly into the existing
speech/trigger extraction pipeline, no new consumer needed.

Progress is resumed by checking the Hub directly for an existing
"<file>.json" sibling, not a local progress file -- a local file ties
resumability to one specific machine, which breaks the moment the job needs
to move (as it did: started on tinkerspace, continuing on RunPod). The Hub
is the single source of truth regardless of which machine is running.

    python3 scripts/transcribe_audios2.py --model large-v3 --compute-type float16 \
        --transcribe-workers 6 --producer-workers 4 --uploader-workers 4
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

REPO_ID = "aoxo/t2a-mommy"
BASE = Path(os.environ.get("TRANSCRIBE_BASE", str(Path.home() / "t2a")))
STAGE_DIR = BASE / "transcribe_stage"
LOG_FILE = BASE / "transcribe_audios2.log"

# A gap this long between two consecutive words becomes its own "silence"
# entry (matching segment.py's own PHRASE_GAP_S=0.7 for what counts as a
# real break, not just natural word spacing).
SILENCE_GAP_S = 0.7

_log_lock = threading.Lock()


def hf_token() -> str:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not set; refusing the machine huggingface cache")
    return token


def hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=hf_token())


def nvidia_mem_mb() -> tuple[int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    total, used = [int(x.strip()) for x in out.split(",")[:2]]
    return total, used


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")


def words_to_alignment(segments) -> list[dict]:
    """Flatten faster-whisper segments into the word+silence entry schema."""
    entries: list[dict] = []
    prev_end = 0.0
    for seg in segments:
        for w in (seg.words or []):
            if w.start - prev_end > SILENCE_GAP_S:
                entries.append({"type": "silence", "start": prev_end, "end": w.start})
            entries.append({
                "type": "word", "start": w.start, "end": w.end,
                "word": w.word.strip(),
            })
            prev_end = w.end
    return entries


def producer(work: queue.Queue, download_q: queue.Queue,
            remaining: list[int], lock: threading.Lock, n_transcribers: int) -> None:
    from huggingface_hub import hf_hub_download

    for rel_path in iter(work.get, None):
        try:
            local = hf_hub_download(REPO_ID, rel_path, repo_type="dataset",
                                    local_dir=STAGE_DIR, token=hf_token())
        except Exception as exc:  # noqa: BLE001 - one bad download must not kill the pipeline
            if "File size mismatch" in str(exc):
                try:
                    local = hf_hub_download(
                        REPO_ID,
                        rel_path,
                        repo_type="dataset",
                        local_dir=STAGE_DIR,
                        token=hf_token(),
                        force_download=True,
                    )
                    download_q.put((rel_path, Path(local)))
                    continue
                except Exception as retry_exc:  # noqa: BLE001
                    # Feed an empty placeholder through the normal permanent
                    # invalid-audio path, producing an empty JSON marker.
                    placeholder = STAGE_DIR / rel_path
                    placeholder.parent.mkdir(parents=True, exist_ok=True)
                    placeholder.write_bytes(b"")
                    download_q.put((rel_path, placeholder))
                    log(f"  [producer] permanently unreadable {rel_path}: "
                        f"{retry_exc}")
                    continue
            log(f"  [producer] download failed for {rel_path}: {exc}")
            continue
        download_q.put((rel_path, Path(local)))
    # With multiple producers, each pushing its own sentinel would leave the
    # count mismatched against however many transcribers are waiting (too
    # few sentinels hangs the extra transcribers forever; this isn't a
    # theoretical concern, it's what a naive one-sentinel-per-thread version
    # of this function actually did). Whichever producer finishes last posts
    # exactly n_transcribers sentinels, which is the number that's actually
    # needed regardless of how many producers there were.
    with lock:
        remaining[0] -= 1
        if remaining[0] == 0:
            for _ in range(n_transcribers):
                download_q.put(None)


MIN_SPEECH_RATIO = float(os.environ.get("T2A_MIN_SPEECH_RATIO", "0.05"))
BATCH_SIZE = int(os.environ.get("T2A_WHISPER_BATCH", "16"))
_PIPES: dict = {}
MIN_SPEECH_SECONDS = float(os.environ.get("T2A_MIN_SPEECH_SECONDS", "20"))


def speech_stats(audio, sample_rate: int, vad_min_silence_ms: int) -> tuple[float, float]:
    """(speech_seconds, total_seconds) via the same Silero VAD faster-whisper uses."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    total = len(audio) / sample_rate
    spans = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=vad_min_silence_ms))
    return sum((sp["end"] - sp["start"]) for sp in spans) / sample_rate, total


def transcriber(model, download_q: queue.Queue, upload_q: queue.Queue, worker_id: int,
                remaining: list[int], lock: threading.Lock, n_uploaders: int,
                vad_filter: bool, vad_min_silence_ms: int) -> None:
    from faster_whisper.audio import decode_audio
    for rel_path, local_path in iter(download_q.get, None):
        json_path = local_path.with_suffix(local_path.suffix + ".json")
        if json_path.exists():
            upload_q.put((rel_path, json_path))
            continue
        try:
            # Gate on speech content first: a file that is almost all silence/room tone
            # costs download + decode + VAD only, never a Whisper pass.
            audio = decode_audio(str(local_path), sampling_rate=16000)
            speech_s, total_s = speech_stats(audio, 16000, vad_min_silence_ms)
            if total_s > 0 and (speech_s / total_s < MIN_SPEECH_RATIO or speech_s < MIN_SPEECH_SECONDS):
                json_path.write_text("[]")
                local_path.unlink(missing_ok=True)
                upload_q.put((rel_path, json_path))
                log(f"  [transcriber-{worker_id}] skipped (speech {speech_s:.0f}s / {total_s:.0f}s = "
                    f"{100*speech_s/max(total_s,1e-6):.1f}%) {rel_path}")
                continue
            # ASMR audio structurally has long non-speech stretches (trigger
            # sounds with no talking over them) mixed with speech -- VAD
            # skips those before they ever reach the decoder, rather than
            # running full Whisper inference over audio with nothing to
            # transcribe. This doesn't lose the gap information: the caller
            # only ever wanted word-level timing for speech anyway, and
            # words_to_alignment() already turns any gap between consecutive
            # words into a "silence" entry -- a VAD-skipped stretch is just a
            # (typically much larger) instance of that same gap, needing no
            # separate handling.
            if BATCH_SIZE > 0:
                from faster_whisper import BatchedInferencePipeline
                pipe = _PIPES.get(id(model)) or _PIPES.setdefault(id(model), BatchedInferencePipeline(model=model))
                segments, _info = pipe.transcribe(audio, batch_size=BATCH_SIZE, word_timestamps=True, vad_filter=vad_filter,
                                                  vad_parameters={"min_silence_duration_ms": vad_min_silence_ms})
            else:
                segments, _info = model.transcribe(
                    audio, word_timestamps=True,
                    vad_filter=vad_filter,
                    vad_parameters={"min_silence_duration_ms": vad_min_silence_ms},
                )
            entries = words_to_alignment(list(segments))
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the pipeline
            if "Invalid data found when processing input" in str(exc):
                # The source object is not decodable audio. Record a valid
                # empty alignment so this permanently bad file does not get
                # retried forever and block fleet completion.
                json_path = local_path.with_suffix(local_path.suffix + ".json")
                json_path.write_text("[]")
                local_path.unlink(missing_ok=True)
                upload_q.put((rel_path, json_path))
                log(f"  [transcriber-{worker_id}] permanently invalid {rel_path}: "
                    "uploaded empty alignment")
                continue
            log(f"  [transcriber-{worker_id}] failed for {rel_path}: {exc}")
            local_path.unlink(missing_ok=True)
            continue

        json_path = local_path.with_suffix(local_path.suffix + ".json")
        import json
        json_path.write_text(json.dumps(entries))
        log(f"  [transcriber-{worker_id}] {rel_path}: {len(entries)} entries (speech {100*speech_s/max(total_s,1e-6):.0f}%)")
        local_path.unlink(missing_ok=True)  # audio already lives on the Hub; done with our copy
        upload_q.put((rel_path, json_path))  # uploader batches to 128-file Hub commits
    with lock:
        remaining[0] -= 1
        if remaining[0] == 0:
            for _ in range(n_uploaders):
                upload_q.put(None)


UPLOAD_BATCH_SIZE = 128
# Fill a 128-file commit before hitting the Hub. ~128 commits/hour is the
# cap; one commit per 128 transcripts stays far under it even with several
# GPUs. Flush a partial batch only after this timeout so the last files of
# a shard still land.
UPLOAD_BATCH_TIMEOUT_S = 45 * 60


def _commit_batch(api, batch: list[tuple[str, Path]]) -> bool:
    """One create_commit() call for the whole batch, instead of one commit
    per file. The Hub's commit rate limit (128/hour, hit in practice with 16
    parallel transcribers each pushing single-file commits) is per-commit,
    not per-file inside a commit -- batching is the actual fix, not just a
    speed optimization.
    """
    from huggingface_hub import CommitOperationAdd

    try:
        # A duplicate/restarted process may have already committed and cleaned
        # one of these paths. Treat that as completed instead of letting
        # CommitOperationAdd raise outside the retry handler and kill the
        # uploader thread.
        existing = [(rel_path, json_path) for rel_path, json_path in batch
                    if json_path.exists()]
        if not existing:
            return True
        ops = [
            CommitOperationAdd(
                path_in_repo=rel_path + ".json",
                path_or_fileobj=str(json_path),
            )
            for rel_path, json_path in existing
        ]
        api.create_commit(
            repo_id=REPO_ID, repo_type="dataset", operations=ops,
            commit_message=f"transcripts for {len(existing)} files",
        )
        return True
    except Exception as exc:  # noqa: BLE001 - caller decides whether to retry
        log(f"  [uploader] batch commit failed ({len(batch)} files): {exc}")
        return False


def uploader(upload_q: queue.Queue, done_count: list[int], lock: threading.Lock,
             batch_size: int, batch_timeout_s: float) -> None:
    api = hf_api()

    batch: list[tuple[str, Path]] = []
    batch_deadline: float | None = None
    finished = False

    while not finished:
        timeout = None
        if batch_deadline is not None:
            timeout = max(0.0, batch_deadline - time.time())
        try:
            item = upload_q.get(timeout=timeout)
        except queue.Empty:
            item = "FLUSH"  # deadline hit with a non-empty batch

        if item is None:
            finished = True
        elif item != "FLUSH":
            batch.append(item)
            if batch_deadline is None:
                batch_deadline = time.time() + batch_timeout_s

        should_flush = (
            len(batch) >= batch_size
            or (finished and batch)
            or (item == "FLUSH" and batch)
        )
        if not should_flush:
            continue

        # A failed commit must not lose the (expensive: GPU minutes of
        # transcription) work it represents -- keep retrying with backoff
        # rather than discarding, matching the same instinct as the rest of
        # this pipeline's "never delete on failure" try/except blocks.
        delay = 5.0
        while not _commit_batch(api, batch):
            time.sleep(delay)
            delay = min(delay * 2, 120.0)

        for rel_path, json_path in batch:
            try:
                json_path.unlink(missing_ok=True)
            except OSError as exc:  # noqa: BLE001 - the commit already succeeded
                log(f"  [uploader] cleanup failed for {rel_path} (harmless, "
                    f"already committed): {exc}")
        with lock:
            done_count[0] += len(batch)
            count = done_count[0]
        log(f"  [uploader] committed {len(batch)} files -> Hub ({count} total this run)")
        batch = []
        batch_deadline = None


def main() -> int:
    global REPO_ID, STAGE_DIR, LOG_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_ID,
                    help="HF dataset repo to transcribe (default aoxo/t2a-mommy)")
    ap.add_argument("--model", default="medium",
                    help="faster-whisper model size; medium fits ~1-2GB VRAM, "
                        "large-v3 wants a real GPU (~3GB+ per instance)")
    ap.add_argument("--compute-type", default="int8_float16",
                    help="int8_float16 for VRAM-constrained sharing; float16 "
                        "for a dedicated GPU (faster, more accurate)")
    ap.add_argument("--upload-batch-timeout", type=int, default=int(UPLOAD_BATCH_TIMEOUT_S),
                    help="seconds before flushing a partial Hub commit (push leftover files)")
    ap.add_argument("--transcribe-workers", type=int, default=1,
                    help="independent WhisperModel instances; 0 = pack VRAM "
                        "(load one model, then as many more as free memory allows)")
    ap.add_argument("--producer-workers", type=int, default=1,
                    help="download threads; raise alongside --transcribe-"
                        "workers so network fetch doesn't starve the GPU")
    ap.add_argument("--uploader-workers", type=int, default=1,
                    help="keep at 1: Hub commit rate is per-repo")
    ap.add_argument("--upload-batch-size", type=int, default=UPLOAD_BATCH_SIZE,
                    help="JSON files per Hub commit (128 stays under the hourly cap)")
    ap.add_argument("--creators-file", type=Path, default=None,
                    help="only transcribe these top-level folder names (one per line). "
                        "Use to pin creators to a GPU so two machines never share a file")
    ap.add_argument("--exclude-creators-file", type=Path, default=None,
                    help="exclude top-level folder names listed one per line")
    ap.add_argument("--vad-filter", dest="vad_filter", action="store_true", default=True,
                    help="skip non-speech audio before it reaches the decoder "
                        "(default on -- ASMR audio has long non-speech stretches)")
    ap.add_argument("--no-vad-filter", dest="vad_filter", action="store_false")
    ap.add_argument("--vad-min-silence-ms", type=int, default=1000,
                    help="silence must be at least this long to get skipped; "
                        "lower than faster-whisper's own 2000ms default since "
                        "segment.py already treats gaps over 700ms as real breaks")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the remaining-to-do list across this many "
                        "independent pods; each pod's --shard-index picks its "
                        "own slice, so running the same script on N pods "
                        "against the same repo doesn't have them all race for "
                        "the same first files with no coordination")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="which shard this instance processes, 0-indexed")
    ap.add_argument("--shard-key", choices=["file", "creator"], default="file",
                    help="stable hash key for sharding; file balances large "
                        "creator folders across many disposable pods")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    REPO_ID = args.repo
    STAGE_DIR = BASE / f"transcribe_stage_{REPO_ID.replace('/', '_')}"
    LOG_FILE = BASE / f"transcribe_{REPO_ID.replace('/', '_')}.log"

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    token = hf_token()

    n_workers = args.transcribe_workers
    if n_workers <= 0:
        total_mb, used0 = nvidia_mem_mb()
        log(f"auto-pack workers: GPU {used0}/{total_mb} MiB before first model")
        first = WhisperModel(args.model, device="cuda", device_index=0, compute_type=args.compute_type)
        total_mb, used1 = nvidia_mem_mb()
        per = max(used1 - used0, 2800)
        # Decode activations add ~1–2 GiB per concurrent transcribe() on
        # long ASMR files. Auto-pack used to reserve 1 GiB and then OOM /
        # cudaErrorInvalidDevice on the 20 GB Ada once 5 large-v3 copies
        # all started decoding at once.
        n_workers = max(1, (total_mb - used0 - 6144) // per)
        n_workers = min(n_workers, 8)
        log(f"auto-pack: one model uses {per} MiB -> {n_workers} workers "
            f"(6 GiB decode headroom)")
        models = [first]
        for _ in range(n_workers - 1):
            models.append(WhisperModel(args.model, device="cuda", device_index=0, compute_type=args.compute_type))
    else:
        log(f"loading {n_workers}x faster-whisper {args.model} "
            f"({args.compute_type})...")
        models = [WhisperModel(args.model, device="cuda", device_index=0, compute_type=args.compute_type)
                 for _ in range(n_workers)]
    args.transcribe_workers = n_workers
    if args.producer_workers < n_workers:
        args.producer_workers = n_workers
    total_mb, used = nvidia_mem_mb()
    log(f"model(s) loaded  VRAM {used}/{total_mb} MiB  workers={n_workers}  producers={args.producer_workers}")

    api = hf_api()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    all_set = set(all_files)
    audio_files = [f for f in all_files if f.lower().endswith(".m4a")]
    todo = [f for f in audio_files if f + ".json" not in all_set]
    log(f"{len(audio_files)} audio files total, {len(audio_files) - len(todo)} "
        f"already transcribed (per existing .json on the Hub), {len(todo)} remaining")

    if args.creators_file is not None:
        creators = {ln.strip() for ln in args.creators_file.read_text().splitlines() if ln.strip()}
        pre = len(todo)
        todo = [f for f in todo if f.split("/")[0] in creators]
        log(f"creators-file {args.creators_file}: {len(todo)}/{pre} remaining in {len(creators)} folders")

    if args.exclude_creators_file is not None:
        excluded = {ln.strip() for ln in args.exclude_creators_file.read_text().splitlines()
                    if ln.strip()}
        pre = len(todo)
        todo = [f for f in todo if f.split("/")[0] not in excluded]
        log(f"exclude-creators-file {args.exclude_creators_file}: "
            f"{len(todo)}/{pre} remaining after excluding {len(excluded)} folders")

    if args.num_shards > 1:
        # A stable hash of the filename, not a positional index into `todo`:
        # each pod lists the repo independently and at a slightly different
        # moment, so the list's order and length can differ pod-to-pod (a
        # file finishing between two pods' listing calls shifts everything
        # after it in an index-based split). Hashing the filename itself
        # gives every pod the same shard assignment for the same file
        # regardless of list order -- Python's builtin hash() is salted
        # per-process and would silently break this, so this uses zlib's
        # crc32, which is stable across processes and machines.
        import zlib
        pre_shard = len(todo)
        # Every path has exactly one stable owner. File hashing balances large
        # creator folders across many pods; creator hashing keeps folders whole.
        def shard_key(path: str) -> str:
            return path if args.shard_key == "file" else path.split("/")[0]

        todo = [f for f in todo
                if zlib.crc32(shard_key(f).encode()) % args.num_shards == args.shard_index]
        log(f"shard {args.shard_index}/{args.num_shards}: {len(todo)}/{pre_shard} "
            f"of the remaining files ({args.shard_key} hash)")

    work: queue.Queue = queue.Queue()
    # Backpressure sized to roughly one in-flight file per transcriber, so
    # producers don't run far ahead of what the GPU can actually consume.
    download_q: queue.Queue = queue.Queue(maxsize=max(4, args.transcribe_workers * 2))
    upload_q: queue.Queue = queue.Queue()
    done_count = [0]
    count_lock = threading.Lock()
    producers_remaining = [args.producer_workers]
    producers_lock = threading.Lock()
    transcribers_remaining = [args.transcribe_workers]
    transcribers_lock = threading.Lock()

    for f in todo:
        work.put(f)
    for _ in range(args.producer_workers):
        work.put(None)  # one stop signal per producer thread

    threads = (
        [threading.Thread(target=producer,
                          args=(work, download_q, producers_remaining, producers_lock,
                                args.transcribe_workers),
                          daemon=True)
         for _ in range(args.producer_workers)]
        + [threading.Thread(target=transcriber,
                            args=(models[i], download_q, upload_q, i,
                                  transcribers_remaining, transcribers_lock,
                                  args.uploader_workers, args.vad_filter,
                                  args.vad_min_silence_ms),
                            daemon=True)
           for i in range(args.transcribe_workers)]
        + [threading.Thread(target=uploader, args=(upload_q, done_count, count_lock,
                                                   args.upload_batch_size,
                                                   float(args.upload_batch_timeout)),
                            daemon=True)
           for _ in range(args.uploader_workers)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    log(f"done: {done_count[0]} files transcribed and uploaded this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
