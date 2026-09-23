#!/usr/bin/env python3
"""Label aoxo/t2a-mommy non-speech gaps with Gemini into CLAP-training format.

Gemini replaces CLAP as the classifier (reuse / reject / invent ontology policy).
Output rows match what build_datasets.py + LAION-CLAP fine-tune expect:

  file_name, trigger, intensity, caption, text[], source, start, duration, ...

Local API only — no GPU pods. High worker concurrency for throughput.

    source ~/.t2a_env
    python3 scripts/annotate_audios2_gemini.py --workers 64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.retag_triggers_gemini import (  # noqa: E402
    CONTEXT_WORDS,
    build_prompt,
    classify,
    set_model,
)
from text2asmr.data.ontology import BY_KEY, intensity_from_loudness  # noqa: E402
from text2asmr.data.segment import load_alignment, split_alignment  # noqa: E402

REPO = "aoxo/t2a-mommy"
HERE = Path(__file__).resolve().parents[1]
OUT_DIR = HERE / "label_tool"
OUT_FILE = OUT_DIR / "gemini_audios2.jsonl"  # legacy / merged ledger
TRAIN_FILE = OUT_DIR / "clap_train_audios2.jsonl"
STAGE_DIR = Path(os.environ.get("TRANSCRIBE_BASE", str(Path.home() / "t2a"))) / "gemini_audios2"
SILENCE_FLOOR_DB = -50.0


def stage_dir_for(shard_index: int, num_shards: int) -> Path:
    base = Path(os.environ.get("TRANSCRIBE_BASE", str(Path.home() / "t2a")))
    if num_shards <= 1:
        return base / "gemini_audios2"
    return base / f"gemini_audios2_s{shard_index:02d}"


_hf_token_i = 0
_hf_token_lock = threading.Lock()
_hf_req_times: list[float] = []
_hf_req_lock = threading.Lock()
_HF_WINDOW_S = 300.0
_HF_MAX_PER_WINDOW = int(os.environ.get("HF_MAX_REQ_PER_5MIN", "980"))


def hf_tokens() -> list[str]:
    raw = os.environ.get("HF_TOKENS") or ""
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    if toks:
        return toks
    one = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not one:
        raise RuntimeError("HF_TOKEN is not set")
    extra = (os.environ.get("HF_TOKEN_B") or "").strip()
    return [one, extra] if extra and extra != one else [one]


def hf_token() -> str:
    """Rotate across HF_TOKENS / HF_TOKEN + HF_TOKEN_B (Hub 1000 req / 5 min is per token)."""
    global _hf_token_i
    toks = hf_tokens()
    with _hf_token_lock:
        tok = toks[_hf_token_i % len(toks)]
        _hf_token_i += 1
        return tok


def _hf_rate_wait() -> None:
    """Block until a Hub REST slot is free (cap * number of tokens / 5 min)."""
    cap = max(1, _HF_MAX_PER_WINDOW * max(1, len(hf_tokens())))
    while True:
        now = time.time()
        with _hf_req_lock:
            _hf_req_times[:] = [t for t in _hf_req_times if now - t < _HF_WINDOW_S]
            if len(_hf_req_times) < cap:
                _hf_req_times.append(now)
                return
            wait = _HF_WINDOW_S - (now - _hf_req_times[0]) + 0.05
        time.sleep(min(max(wait, 0.05), 5.0))


MEDIA_ROOT = Path(os.environ.get("TRANSCRIBE_BASE", str(Path.home() / "t2a"))) / "audios2"


def _existing_local(filename: str) -> Path | None:
    """Reuse files already on disk — do not hit Hub if we already have them."""
    candidates = [
        MEDIA_ROOT / filename,
        Path(os.environ.get("TRANSCRIBE_BASE", str(Path.home() / "t2a"))) / "gemini_audios2" / filename,
    ]
    base = Path(os.environ.get("TRANSCRIBE_BASE", str(Path.home() / "t2a")))
    for d in sorted(base.glob("gemini_audios2*")):
        candidates.append(d / filename)
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def hub_download(repo: str, filename: str, local_dir: Path) -> Path:
    """Prefer a local copy. Source audio is deleted after clips are cut."""
    hit = _existing_local(filename)
    if hit is not None:
        return hit

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    dest_root = MEDIA_ROOT
    dest_root.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    kwargs = dict(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest_root),
        token=hf_token(),
    )
    try:
        return Path(hf_hub_download(**kwargs, local_files_only=True))
    except (LocalEntryNotFoundError, HfHubHTTPError, OSError):
        pass
    for attempt in range(8):
        try:
            _hf_rate_wait()
            return Path(hf_hub_download(**kwargs, local_files_only=False))
        except HfHubHTTPError as exc:
            last = exc
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code not in (429, 500, 502, 503):
                raise
            wait = min(90, 8 * (attempt + 1))
            print(f"  Hub {code} on {filename[:60]} — sleep {wait}s", flush=True)
            time.sleep(wait)
    raise last  # type: ignore[misc]


def context_from_entries(
    entries: list[dict], start: float, duration: float, n: int = CONTEXT_WORDS
) -> tuple[str, str]:
    words = [e for e in entries if e.get("type") == "word"]
    before = [str(w.get("word", "")).strip() for w in words if float(w.get("end", 0)) <= start][-n:]
    after = [
        str(w.get("word", "")).strip()
        for w in words
        if float(w.get("start", 0)) >= start + duration
    ][:n]
    return " ".join(w for w in before if w), " ".join(w for w in after if w)


def rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-9))


def decode_mono(path: Path, start: float, dur: float, sr: int = 16_000) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr), "-ac", "1", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def cut_flac(path: Path, start: float, dur: float) -> bytes:
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", str(path), "-ac", "1", "-ar", "16000", "-c:a", "flac", "-f", "flac", "-",
    ]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def clap_text_list(label: str, intensity: str) -> list[str]:
    """LAION-CLAP `text` field: primary caption + ontology probes when known."""
    primary = f"ASMR {intensity} {label}, close-mic binaural, no speech"
    texts = [primary]
    if label in BY_KEY:
        texts.extend(BY_KEY[label].probes)
    texts.append(f"The sound of {label}")
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def to_clap_row(base: dict, label: str, labeler: str = "gemini-3.6-flash") -> dict:
    """Mirror build_datasets.py trigger metadata + LAION text[]."""
    intensity, scaled = intensity_from_loudness(float(base["rms_db"]))
    safe_uid = str(base["uid"]).replace("/", "__")
    texts = clap_text_list(label, intensity)
    return {
        "file_name": f"{safe_uid}.flac",
        "trigger": label,
        "intensity": intensity,
        "intensity_scaled": round(scaled, 3),
        "tag": f"[{intensity}][{label}]",
        "caption": texts[0],
        "text": texts,
        "source": base["source"],
        "start": base["start"],
        "duration": base["duration"],
        "uid": base["uid"],
        "rms_db": base["rms_db"],
        "context_before": base.get("context_before", ""),
        "context_after": base.get("context_after", ""),
        "gemini_label": label,
        "labeler": labeler,
    }


def load_done(paths: list[Path]) -> set[str]:
    done = set()
    uid_gz = OUT_DIR / "done_uids.txt.gz"
    uid_txt = OUT_DIR / "done_uids.txt"
    if uid_gz.exists():
        import gzip
        with gzip.open(uid_gz, "rt", encoding="utf-8") as fh:
            for line in fh:
                uid = line.strip()
                if uid:
                    done.add(uid)
    elif uid_txt.exists():
        with uid_txt.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                uid = line.strip()
                if uid:
                    done.add(uid)
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("uid") and row.get("gemini_label") is not None:
                    done.add(row["uid"])
    return done


def ledger_paths() -> list[Path]:
    paths = [OUT_FILE]
    paths.extend(sorted(OUT_DIR.glob("gemini_audios2*.jsonl")))
    # dedupe
    seen = set()
    out = []
    for p in paths:
        rp = p.resolve() if p.exists() else p
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def shard_sources(sources: list[str], num_shards: int, shard_index: int) -> list[str]:
    if num_shards <= 1:
        return sources
    return [
        s for s in sources
        if zlib.crc32(s.encode()) % num_shards == shard_index
    ]


_AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".webm"}


def drop_cache(path: Path) -> None:
    """Delete source audio after clips are cut. Keep alignment JSON."""
    if path is None:
        return
    try:
        if path.suffix.lower() in _AUDIO_SUFFIXES and path.is_file():
            path.unlink()
    except OSError:
        pass


def paired_sources(files: list[str], creators: set[str] | None) -> list[str]:
    available = set(files)
    paired = [
        path for path in files
        if path.endswith(".m4a") and path + ".json" in available
    ]
    if creators:
        paired = [path for path in paired if path.split("/")[0] in creators]
    return sorted(paired)


def load_creators(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining candidates")
    ap.add_argument("--workers", type=int, default=max(32, (os.cpu_count() or 8) * 4))
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--creators-file", type=Path, default=None)
    ap.add_argument("--max-sources", type=int, default=0, help="0 = every paired source")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--file-list", type=Path, default=None,
                    help="precomputed JSON list of repo files (skip Hub list)")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
                    help="Gemini model id (each model has its own daily quota)")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    backend = os.environ.get("GEMINI_BACKEND", "vertex").strip().lower()
    if not args.inventory_only and backend not in ("vertex", "vertex_ai", "gcp") and not api_key:
        raise SystemExit("GEMINI_API_KEY not set (source ~/.t2a_env first)")

    from huggingface_hub import HfApi, hf_hub_download

    creators = load_creators(args.creators_file)
    if args.file_list and args.file_list.exists():
        files = json.loads(args.file_list.read_text())
        print(f"loaded file list ({len(files)}) from {args.file_list}", flush=True)
    else:
        files = HfApi(token=hf_token()).list_repo_files(REPO, repo_type="dataset")
    sources = paired_sources(files, creators)
    sources = shard_sources(sources, args.num_shards, args.shard_index)
    if args.max_sources:
        sources = sources[: args.max_sources]
    print(
        f"shard={args.shard_index}/{args.num_shards} sources={len(sources)} on {REPO}"
        + (f" (creators={len(creators)})" if creators else ""),
        flush=True,
    )

    set_model(args.model)
    model_tag = args.model.replace("gemini-", "").replace("-flash", "")
    out_path = (
        OUT_DIR / f"gemini_audios2.{model_tag}.shard{args.shard_index:02d}.jsonl"
        if args.num_shards > 1
        else OUT_DIR / f"gemini_audios2.{model_tag}.jsonl"
    )
    train_path = (
        OUT_DIR / f"clap_train_audios2.{model_tag}.shard{args.shard_index:02d}.jsonl"
        if args.num_shards > 1
        else OUT_DIR / f"clap_train_audios2.{model_tag}.jsonl"
    )

    done = load_done(ledger_paths())
    print(f"{len(done)} candidates already labeled across ledgers", flush=True)
    print(f"model={args.model} workers={args.workers} out={out_path.name}", flush=True)

    write_lock = threading.Lock()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global STAGE_DIR
    STAGE_DIR = stage_dir_for(args.shard_index, args.num_shards)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    inventory = {"sources": 0, "candidates": 0, "already_done": 0, "silent": 0}
    processed = 0
    errors = 0
    kept = 0

    def classify_one(item: dict) -> dict:
        label, error = classify(api_key, item["audio"], item["prompt"], model=args.model)
        row = {k: item[k] for k in ("uid", "source", "start", "duration",
                                    "context_before", "context_after", "rms_db")}
        row["gemini_label"] = label
        row["error"] = error
        row["skip_reason"] = None
        row["labeler"] = args.model
        if label and label != "reject":
            row.update(to_clap_row(row, label, labeler=args.model))
        return row

    out = None if args.inventory_only else out_path.open("a")
    train = None if args.inventory_only else train_path.open("a")
    try:
        for i, source in enumerate(sources, 1):
            if args.limit and processed >= args.limit:
                break
            try:
                jp = hub_download(REPO, f"{source}.json", STAGE_DIR)
                entries = load_alignment(jp)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{source}] alignment skip: {type(exc).__name__}: {exc}", flush=True)
                continue

            cands = [s for s in split_alignment(entries, source) if s.kind == "trigger_candidate"]
            inventory["sources"] += 1
            inventory["candidates"] += len(cands)
            pending = [s for s in cands if s.uid not in done]
            inventory["already_done"] += len(cands) - len(pending)
            if args.inventory_only or not pending:
                continue
            if args.limit:
                pending = pending[: args.limit - processed]
            if not pending:
                continue

            try:
                audio_path = hub_download(REPO, source, STAGE_DIR)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{source}] audio skip: {type(exc).__name__}: {exc}", flush=True)
                continue

            batch: list[dict] = []
            silent_rows: list[dict] = []
            for span in pending:
                try:
                    mono = decode_mono(audio_path, span.start, span.duration)
                except subprocess.CalledProcessError:
                    continue
                level = rms_db(mono)
                before, after = context_from_entries(entries, span.start, span.duration)
                base = {
                    "uid": span.uid,
                    "source": source,
                    "start": round(span.start, 3),
                    "duration": round(span.duration, 3),
                    "context_before": before,
                    "context_after": after,
                    "rms_db": round(level, 2),
                }
                if mono.size == 0 or level < SILENCE_FLOOR_DB:
                    inventory["silent"] += 1
                    silent_rows.append({
                        **base,
                        "gemini_label": "reject",
                        "error": None,
                        "skip_reason": "silence_floor",
                    })
                    continue
                try:
                    flac = cut_flac(audio_path, span.start, span.duration)
                except subprocess.CalledProcessError as exc:
                    silent_rows.append({
                        **base,
                        "gemini_label": None,
                        "error": f"ffmpeg: {exc.stderr[-200:] if exc.stderr else exc}",
                        "skip_reason": None,
                    })
                    continue
                batch.append({
                    **base,
                    "audio": flac,
                    "prompt": build_prompt(before, after),
                })

            drop_cache(audio_path)

            for row in silent_rows:
                out.write(json.dumps(row) + "\n")
                done.add(row["uid"])
                processed += 1
            if silent_rows:
                out.flush()

            if batch:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futures = [ex.submit(classify_one, item) for item in batch]
                    for fut in as_completed(futures):
                        row = fut.result()
                        with write_lock:
                            out.write(json.dumps(row) + "\n")
                            out.flush()
                            if row.get("trigger") and row.get("gemini_label") not in (None, "reject"):
                                train.write(json.dumps({
                                    k: row[k] for k in (
                                        "file_name", "trigger", "intensity", "intensity_scaled",
                                        "tag", "caption", "text", "source", "start", "duration",
                                        "uid", "rms_db", "labeler",
                                    ) if k in row
                                }) + "\n")
                                train.flush()
                                kept += 1
                        done.add(row["uid"])
                        processed += 1
                        if row.get("gemini_label") is None:
                            errors += 1

            if i % 5 == 0 or processed:
                print(
                    f"[shard{args.shard_index:02d} {i}/{len(sources)}] "
                    f"labeled={processed} kept={kept} errors={errors} "
                    f"cands_seen={inventory['candidates']}",
                    flush=True,
                )
            if args.limit and processed >= args.limit:
                break
    finally:
        if out is not None:
            out.close()
        if train is not None:
            train.close()

    print(
        f"inventory sources={inventory['sources']} candidates={inventory['candidates']} "
        f"already_done={inventory['already_done']} silent={inventory['silent']}",
        flush=True,
    )
    if not args.inventory_only:
        print(
            f"{processed} rows ({kept} clap-train, {errors} errors) -> "
            f"{out_path} + {train_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
