#!/usr/bin/env python3
"""Transcribe aoxo/audios2 into word-level alignments, for training v2.

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

Alignment JSON matches aoxo/audios's own schema (a flat list of
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
import queue
import shutil
import threading
import time
from pathlib import Path

REPO_ID = "aoxo/audios2"
BASE = Path.home() / "t2a"
STAGE_DIR = BASE / "transcribe_stage"
LOG_FILE = BASE / "transcribe_audios2.log"

# A gap this long between two consecutive words becomes its own "silence"
# entry (matching segment.py's own PHRASE_GAP_S=0.7 for what counts as a
# real break, not just natural word spacing).
SILENCE_GAP_S = 0.7

_log_lock = threading.Lock()


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
                                    local_dir=STAGE_DIR)
        except Exception as exc:  # noqa: BLE001 - one bad download must not kill the pipeline
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


def transcriber(model, download_q: queue.Queue, upload_q: queue.Queue, worker_id: int,
                remaining: list[int], lock: threading.Lock, n_uploaders: int,
                vad_filter: bool, vad_min_silence_ms: int) -> None:
    for rel_path, local_path in iter(download_q.get, None):
        try:
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
            segments, _info = model.transcribe(
                str(local_path), word_timestamps=True,
                vad_filter=vad_filter,
                vad_parameters={"min_silence_duration_ms": vad_min_silence_ms},
            )
            entries = words_to_alignment(list(segments))
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the pipeline
            log(f"  [transcriber-{worker_id}] failed for {rel_path}: {exc}")
            local_path.unlink(missing_ok=True)
            continue

        json_path = local_path.with_suffix(local_path.suffix + ".json")
        import json
        json_path.write_text(json.dumps(entries))
        local_path.unlink(missing_ok=True)  # audio already lives on the Hub; done with our copy
        upload_q.put((rel_path, json_path))
        log(f"  [transcriber-{worker_id}] {rel_path}: {len(entries)} entries")
    with lock:
        remaining[0] -= 1
        if remaining[0] == 0:
            for _ in range(n_uploaders):
                upload_q.put(None)


UPLOAD_BATCH_SIZE = 20
# 30s was still landing close to the Hub's 128/hour commit ceiling in
# practice: real throughput (~90-105 files/hour across 16 transcribers) means
# most batches flush via this timeout with only 1-4 files rather than
# reaching UPLOAD_BATCH_SIZE, so the steady-state commit rate was ~120/hour
# with no margin for bursts -- confirmed by 266 rate-limit hits in one run
# (survived via retry, but still real wasted time). 90s drops steady-state
# to ~40/hour, comfortable headroom even during a burst of several files
# finishing at once.
UPLOAD_BATCH_TIMEOUT_S = 90.0


def _commit_batch(api, batch: list[tuple[str, Path]]) -> bool:
    """One create_commit() call for the whole batch, instead of one commit
    per file. The Hub's commit rate limit (128/hour, hit in practice with 16
    parallel transcribers each pushing single-file commits) is per-commit,
    not per-file inside a commit -- batching is the actual fix, not just a
    speed optimization.
    """
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(path_in_repo=rel_path + ".json", path_or_fileobj=str(json_path))
          for rel_path, json_path in batch]
    try:
        api.create_commit(
            repo_id=REPO_ID, repo_type="dataset", operations=ops,
            commit_message=f"transcripts for {len(batch)} files",
        )
        return True
    except Exception as exc:  # noqa: BLE001 - caller decides whether to retry
        log(f"  [uploader] batch commit failed ({len(batch)} files): {exc}")
        return False


def uploader(upload_q: queue.Queue, done_count: list[int], lock: threading.Lock) -> None:
    from huggingface_hub import HfApi
    api = HfApi()

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
                batch_deadline = time.time() + UPLOAD_BATCH_TIMEOUT_S

        should_flush = (
            len(batch) >= UPLOAD_BATCH_SIZE
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium",
                    help="faster-whisper model size; medium fits ~1-2GB VRAM, "
                        "large-v3 wants a real GPU (~3GB+ per instance)")
    ap.add_argument("--compute-type", default="int8_float16",
                    help="int8_float16 for VRAM-constrained sharing; float16 "
                        "for a dedicated GPU (faster, more accurate)")
    ap.add_argument("--transcribe-workers", type=int, default=1,
                    help="independent WhisperModel instances -- one faster-"
                        "whisper call doesn't saturate a big GPU's FLOPs, so "
                        "this is the real parallelism knob on a dedicated card")
    ap.add_argument("--producer-workers", type=int, default=1,
                    help="download threads; raise alongside --transcribe-"
                        "workers so network fetch doesn't starve the GPU")
    ap.add_argument("--uploader-workers", type=int, default=1,
                    help="the Hub's commit rate limit (128/hour, hit in "
                        "practice at --transcribe-workers 16 before batching "
                        "was added) is per-repo, not per-connection -- more "
                        "uploader threads each batching independently just "
                        "multiplies commit frequency without raising the "
                        "ceiling, so 1 is correct here, not a bottleneck")
    ap.add_argument("--vad-filter", dest="vad_filter", action="store_true", default=True,
                    help="skip non-speech audio before it reaches the decoder "
                        "(default on -- ASMR audio has long non-speech stretches)")
    ap.add_argument("--no-vad-filter", dest="vad_filter", action="store_false")
    ap.add_argument("--vad-min-silence-ms", type=int, default=1000,
                    help="silence must be at least this long to get skipped; "
                        "lower than faster-whisper's own 2000ms default since "
                        "segment.py already treats gaps over 700ms as real breaks")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from huggingface_hub import HfApi

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"loading {args.transcribe_workers}x faster-whisper {args.model} "
        f"({args.compute_type})...")
    models = [WhisperModel(args.model, device="cuda", compute_type=args.compute_type)
             for _ in range(args.transcribe_workers)]
    log("model(s) loaded")

    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    all_set = set(all_files)
    audio_files = [f for f in all_files if f.lower().endswith(".m4a")]
    todo = [f for f in audio_files if f + ".json" not in all_set]
    log(f"{len(audio_files)} audio files total, {len(audio_files) - len(todo)} "
        f"already transcribed (per existing .json on the Hub), {len(todo)} remaining")

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
        + [threading.Thread(target=uploader, args=(upload_q, done_count, count_lock),
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
