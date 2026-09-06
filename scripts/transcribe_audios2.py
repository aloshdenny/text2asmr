#!/usr/bin/env python3
"""Transcribe aoxo/audios2 into word-level alignments, for training v2.

Three concurrent stages, each bottlenecked on a different resource, so none
of them sit idle waiting on the others:

  producer (network)  -> download_q -> transcriber (GPU) -> upload_q -> uploader (network)

The transcriber is the only stage touching the GPU, and runs a single
faster-whisper model instance (~1-2GB VRAM) so it coexists with the trigger
LoRA training already running on this card (~11-13GB) without contention --
verified empirically before this script existed by loading the model
standalone and checking nvidia-smi.

Alignment JSON matches aoxo/audios's own schema (a flat list of
{"type": "word"|"silence", "start", "end", ...}), the same shape segment.py
already parses -- so this output slots directly into the existing
speech/trigger extraction pipeline, no new consumer needed.

Progress is tracked in transcribed.txt (one "creator/filename" per line, one
level of granularity finer than done.txt's per-creator tracking, since
transcription happens per file). A file is only added to it after its JSON
is confirmed uploaded -- a killed process re-lists the repo and just skips
what's already marked done, safe to resume anytime.

    python3 scripts/transcribe_audios2.py --model medium --workers 1
"""

from __future__ import annotations

import argparse
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

REPO_ID = "aoxo/audios2"
BASE = Path.home() / "t2a"
STAGE_DIR = BASE / "transcribe_stage"
TRANSCRIBED_FILE = BASE / "transcribed.txt"
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


def read_done() -> set[str]:
    if not TRANSCRIBED_FILE.exists():
        return set()
    return {ln.strip() for ln in TRANSCRIBED_FILE.read_text().splitlines() if ln.strip()}


def mark_done(rel_path: str) -> None:
    with TRANSCRIBED_FILE.open("a") as f:
        f.write(rel_path + "\n")


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


def producer(work: queue.Queue, download_q: queue.Queue, stop: threading.Event) -> None:
    from huggingface_hub import hf_hub_download

    for rel_path in iter(work.get, None):
        if stop.is_set():
            break
        try:
            local = hf_hub_download(REPO_ID, rel_path, repo_type="dataset",
                                    local_dir=STAGE_DIR)
        except Exception as exc:  # noqa: BLE001 - one bad download must not kill the pipeline
            log(f"  [producer] download failed for {rel_path}: {exc}")
            continue
        download_q.put((rel_path, Path(local)))
    download_q.put(None)  # sentinel: no more work coming


def transcriber(model, download_q: queue.Queue, upload_q: queue.Queue) -> None:
    for rel_path, local_path in iter(download_q.get, None):
        try:
            segments, _info = model.transcribe(str(local_path), word_timestamps=True)
            entries = words_to_alignment(list(segments))
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the pipeline
            log(f"  [transcriber] failed for {rel_path}: {exc}")
            local_path.unlink(missing_ok=True)
            continue

        json_path = local_path.with_suffix(local_path.suffix + ".json")
        import json
        json_path.write_text(json.dumps(entries))
        local_path.unlink(missing_ok=True)  # audio already lives on the Hub; done with our copy
        upload_q.put((rel_path, json_path))
        log(f"  [transcriber] {rel_path}: {len(entries)} entries")
    upload_q.put(None)


def uploader(upload_q: queue.Queue, done_count: list[int]) -> None:
    from huggingface_hub import HfApi
    api = HfApi()

    for rel_path, json_path in iter(upload_q.get, None):
        remote_path = rel_path + ".json"
        try:
            api.upload_file(
                path_or_fileobj=str(json_path), path_in_repo=remote_path,
                repo_id=REPO_ID, repo_type="dataset",
                commit_message=f"transcript for {rel_path}",
            )
        except Exception as exc:  # noqa: BLE001 - one bad upload must not kill the pipeline
            log(f"  [uploader] upload failed for {rel_path}: {exc}")
            json_path.unlink(missing_ok=True)
            continue
        mark_done(rel_path)
        json_path.unlink(missing_ok=True)
        done_count[0] += 1
        log(f"  [uploader] {rel_path} -> transcribed.txt ({done_count[0]} total this run)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium",
                    help="faster-whisper model size; medium fits ~1-2GB VRAM")
    ap.add_argument("--compute-type", default="int8_float16")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from huggingface_hub import HfApi

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"loading faster-whisper {args.model} ({args.compute_type})...")
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)
    log("model loaded")

    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    audio_files = [f for f in all_files if f.lower().endswith(".m4a")]
    done = read_done()
    todo = [f for f in audio_files if f not in done]
    log(f"{len(audio_files)} audio files total, {len(done)} already transcribed, "
        f"{len(todo)} remaining")

    work: queue.Queue = queue.Queue()
    download_q: queue.Queue = queue.Queue(maxsize=4)  # backpressure: don't outrun the GPU
    upload_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    done_count = [0]

    for f in todo:
        work.put(f)
    work.put(None)

    threads = [
        threading.Thread(target=producer, args=(work, download_q, stop), daemon=True),
        threading.Thread(target=transcriber, args=(model, download_q, upload_q), daemon=True),
        threading.Thread(target=uploader, args=(upload_q, done_count), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    log(f"done: {done_count[0]} files transcribed and uploaded this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
