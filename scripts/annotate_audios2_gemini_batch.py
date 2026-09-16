#!/usr/bin/env python3
"""Vertex Batch API labeling for aoxo/audios2 non-speech gaps.

Vertex does not accept inline Batch requests — clips go to GCS, then
batchPredictionJobs (~50% vs unary). Model fallback: 3.8-flash → 3.7 → 3.6.

    export GEMINI_BACKEND=vertex
    source ~/.t2a_env   # HF_TOKEN for Hub downloads
    python3 scripts/annotate_audios2_gemini_batch.py --poll --until-empty
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from scripts.annotate_audios2_gemini import (  # noqa: E402
    OUT_DIR,
    REPO,
    SILENCE_FLOOR_DB,
    context_from_entries,
    cut_flac,
    decode_mono,
    drop_cache,
    hub_download,
    ledger_paths,
    load_done,
    paired_sources,
    rms_db,
    shard_sources,
    stage_dir_for,
    to_clap_row,
)
from scripts.retag_triggers_gemini import build_prompt  # noqa: E402
from text2asmr.data.segment import load_alignment, split_alignment  # noqa: E402

JOBS_FILE = OUT_DIR / "gemini_batch_jobs.jsonl"
VERTEX_PROJECT = os.environ.get("GEMINI_VERTEX_PROJECT", "project-9da5a2fe-3df4-485e-9a9")
VERTEX_LOCATION = os.environ.get("GEMINI_VERTEX_LOCATION", "global")
GCS_PREFIX = os.environ.get(
    "GEMINI_BATCH_GCS", "gs://aleddo-splitter-eval/t2a-gemini-batch"
)
FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_BATCH_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash"
    ).split(",")
    if m.strip()
]
# Concurrent Hub downloads. REST is capped in hub_download (980 req / 5 min / token);
# this semaphore is the bandwidth side (many in-flight xet/m4a transfers).
HUB_SEM = threading.Semaphore(int(os.environ.get("GEMINI_HUB_SEM", "64")))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_gcloud_token() -> None:
    from google.auth.transport.requests import Request
    import google.auth
    from google.auth.exceptions import TransportError

    last: Exception | None = None
    for attempt in range(8):
        try:
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(Request())
            os.environ["CLOUDSDK_AUTH_ACCESS_TOKEN"] = creds.token
            os.environ.setdefault("CLOUDSDK_CORE_DISABLE_FILE_LOGGING", "1")
            return
        except (TransportError, OSError) as exc:
            last = exc
            wait = min(60, 4 * (attempt + 1))
            log(f"gcloud token refresh failed ({type(exc).__name__}); retry in {wait}s")
            time.sleep(wait)
    raise last  # type: ignore[misc]


def client():
    from google import genai

    return genai.Client(
        vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION
    )


def parse_label_from_response(resp) -> tuple[str | None, str | None]:
    try:
        if isinstance(resp, dict):
            feedback = resp.get("promptFeedback") or resp.get("prompt_feedback") or {}
            if (
                feedback.get("blockReason") == "PROHIBITED_CONTENT"
                or feedback.get("block_reason") == "PROHIBITED_CONTENT"
            ):
                return "reject", None
            cands = resp.get("candidates") or []
            if not cands:
                return None, f"no candidates: {str(resp)[:240]}"
            finish = (cands[0] or {}).get("finishReason") or (cands[0] or {}).get("finish_reason")
            parts = (((cands[0] or {}).get("content") or {}).get("parts") or [])
            if finish == "SAFETY" or not parts:
                return "reject", None
            text = (parts[0] or {}).get("text") if parts else None
        else:
            feedback = getattr(resp, "prompt_feedback", None)
            if feedback is not None and getattr(feedback, "block_reason", None):
                if "PROHIBITED" in str(feedback.block_reason):
                    return "reject", None
            text = getattr(resp, "text", None)
            if text is None and getattr(resp, "candidates", None):
                parts = resp.candidates[0].content.parts
                text = parts[0].text
        if not text:
            return None, "empty text"
        return json.loads(text)["label"].strip().lower(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"parse failed: {exc}"


def _process_source(source: str, done: set[str], done_lock: threading.Lock, stage: Path) -> list[dict]:
    with done_lock:
        # cheap skip if we somehow already reserved — still parse alignment
        pass
    try:
        with HUB_SEM:
            jp = hub_download(REPO, f"{source}.json", stage)
        entries = load_alignment(jp)
    except Exception as exc:  # noqa: BLE001
        log(f"align skip {source}: {type(exc).__name__}: {exc}")
        return []
    cands = [s for s in split_alignment(entries, source) if s.kind == "trigger_candidate"]
    with done_lock:
        pending = [s for s in cands if s.uid not in done]
        for s in pending:
            done.add(s.uid)  # reserve
    if not pending:
        return []
    try:
        with HUB_SEM:
            audio_path = hub_download(REPO, source, stage)
    except Exception as exc:  # noqa: BLE001
        log(f"audio skip {source}: {type(exc).__name__}: {exc}")
        with done_lock:
            for s in pending:
                done.discard(s.uid)
        return []

    out: list[dict] = []
    try:
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
                "rms_db": round(float(level), 2),
            }
            if mono.size == 0 or level < SILENCE_FLOOR_DB:
                base.update({
                    "gemini_label": "reject",
                    "error": None,
                    "skip_reason": "silence_floor",
                    "labeler": "local-silence",
                    "_local_reject": True,
                })
                out.append(base)
                continue
            try:
                flac = cut_flac(audio_path, span.start, span.duration)
            except subprocess.CalledProcessError:
                continue
            out.append({
                **base,
                "prompt": build_prompt(before, after),
                "flac": flac,
                "_local_reject": False,
            })
    finally:
        drop_cache(audio_path)
    return out


def collect_candidates(
    *,
    files: list[str],
    done: set[str],
    num_shards: int,
    shard_index: int,
    limit: int,
    stage: Path,
    workers: int,
    on_api_batch=None,
    api_batch_size: int = 0,
    on_local_reject=None,
) -> list[dict]:
    sources = shard_sources(paired_sources(files, None), num_shards, shard_index)
    out: list[dict] = []
    done_lock = threading.Lock()
    src_iter = iter(sources)
    inflight: dict = {}
    workers = max(1, workers)
    api_emitted = 0

    def _flush_callbacks(force: bool = False) -> None:
        nonlocal api_emitted
        if on_local_reject:
            rejs = [x for x in out if x.get("_local_reject")]
            if rejs:
                on_local_reject(rejs)
                out[:] = [x for x in out if not x.get("_local_reject")]
        if not on_api_batch or not api_batch_size:
            return
        while True:
            apis = [x for x in out if not x.get("_local_reject")]
            take = len(apis) if force else (api_batch_size if len(apis) >= api_batch_size else 0)
            if not take:
                return
            chunk, kept_api, kept_other = [], 0, []
            for row in out:
                if row.get("_local_reject"):
                    kept_other.append(row)
                    continue
                if kept_api < take:
                    chunk.append(row)
                    kept_api += 1
                else:
                    kept_other.append(row)
            out[:] = kept_other
            if chunk:
                on_api_batch(chunk)
                api_emitted += len(chunk)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        def _fill() -> None:
            while len(inflight) < workers:
                try:
                    src = next(src_iter)
                except StopIteration:
                    return
                inflight[pool.submit(_process_source, src, done, done_lock, stage)] = src

        _fill()
        while inflight:
            fut = next(as_completed(list(inflight)))
            src = inflight.pop(fut)
            try:
                rows = fut.result()
            except Exception as exc:  # noqa: BLE001
                log(f"source fail {src}: {exc}")
                _fill()
                continue
            out.extend(rows)
            _flush_callbacks(force=False)
            api_n = api_emitted + sum(1 for x in out if not x.get("_local_reject"))
            if limit and api_n >= limit:
                for leftover in inflight:
                    leftover.cancel()
                inflight.clear()
                break
            _fill()
    _flush_callbacks(force=False)
    api_n = api_emitted + sum(1 for x in out if not x.get("_local_reject"))
    if limit and api_n > limit:
        kept: list[dict] = []
        api = api_emitted
        for row in out:
            if row.get("_local_reject"):
                kept.append(row)
                continue
            if api >= limit:
                with done_lock:
                    done.discard(row["uid"])
                continue
            kept.append(row)
            api += 1
        out = kept
    _flush_callbacks(force=True)
    return out


def write_row(out_f, train_f, row: dict, model: str) -> None:
    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_f.flush()
    lab = row.get("gemini_label")
    if lab and lab != "reject" and not row.get("error"):
        clap = to_clap_row(row, lab, labeler=f"batch:{model}")
        train_f.write(json.dumps(clap, ensure_ascii=False) + "\n")
        train_f.flush()


def _split_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(uri)
    rest = uri[5:]
    bucket, _, path = rest.partition("/")
    return bucket, path


def _gcs_client():
    from google.cloud import storage

    return storage.Client(project=VERTEX_PROJECT)


def _gcs_cp(*args: str) -> None:
    src, dest = args[0], args[-1]
    try:
        client = _gcs_client()
        if src.startswith("gs://") and not dest.startswith("gs://"):
            b, p = _split_gs(src)
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            client.bucket(b).blob(p).download_to_filename(dest)
            return
        if dest.startswith("gs://") and not src.startswith("gs://"):
            b, p = _split_gs(dest)
            client.bucket(b).blob(p).upload_from_filename(src)
            return
    except Exception as exc:
        if not shutil.which("gcloud"):
            raise
        log(f"python gcs cp failed ({type(exc).__name__}); trying gcloud")
    ensure_gcloud_token()
    subprocess.run(["gcloud", "storage", "cp", src, dest], check=True, capture_output=True)


def _gcs_cp_recursive(src: str, dest: str) -> None:
    try:
        client = _gcs_client()
        if dest.startswith("gs://") and not src.startswith("gs://"):
            b, prefix = _split_gs(dest.rstrip("/"))
            bucket = client.bucket(b)
            src_p = Path(src)
            for f in src_p.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(src_p).as_posix()
                blob = f"{prefix}/{rel}" if prefix else rel
                bucket.blob(blob).upload_from_filename(str(f))
            return
        if src.startswith("gs://") and not dest.startswith("gs://"):
            b, prefix = _split_gs(src.rstrip("/").replace("/**", ""))
            dest_p = Path(dest)
            dest_p.mkdir(parents=True, exist_ok=True)
            for blob in client.list_blobs(b, prefix=prefix):
                if blob.name.endswith("/"):
                    continue
                out = dest_p / Path(blob.name).name
                blob.download_to_filename(str(out))
            return
    except Exception as exc:
        if not shutil.which("gcloud"):
            raise
        log(f"python gcs recursive failed ({type(exc).__name__}); trying gcloud")
    ensure_gcloud_token()
    subprocess.run(
        ["gcloud", "storage", "cp", "--recursive", src, dest],
        check=True,
        capture_output=True,
    )


def _gcs_ls(prefix: str) -> list[str]:
    try:
        client = _gcs_client()
        raw = prefix.replace("/**", "").rstrip("/")
        b, p = _split_gs(raw)
        out = []
        for blob in client.list_blobs(b, prefix=(p + "/") if p else ""):
            if blob.name.endswith("/"):
                continue
            out.append(f"gs://{b}/{blob.name}")
        return out
    except Exception:
        if not shutil.which("gcloud"):
            return []
    ensure_gcloud_token()
    r = subprocess.run(
        ["gcloud", "storage", "ls", prefix],
        check=False,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return []
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def submit_batch(cli, model: str, items: list[dict], work: Path) -> str:
    """Upload FLACs + JSONL to GCS and create a Vertex batchPredictionJob."""
    work.mkdir(parents=True, exist_ok=True)
    flac_dir = work / "flac"
    flac_dir.mkdir(exist_ok=True)
    key_meta = {}
    lines = []
    stamp = int(time.time())
    gcs_job = f"{GCS_PREFIX}/{model.replace('/', '_')}_{stamp}_{os.getpid()}"
    for i, item in enumerate(items):
        uid = item["uid"]
        key = f"u{i:05d}_{zlib.crc32(uid.encode()) & 0xffffffff:08x}"
        local = flac_dir / f"{key}.flac"
        local.write_bytes(item["flac"])
        gcs_flac = f"{gcs_job}/flac/{key}.flac"
        key_meta[key] = {
            **{k: item[k] for k in (
                "uid", "source", "start", "duration",
                "context_before", "context_after", "rms_db",
            )},
            "__index": i,
            "gcs_flac": gcs_flac,
        }
        lines.append(json.dumps({
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"fileData": {"fileUri": gcs_flac, "mimeType": "audio/flac"}},
                        {"text": item["prompt"]},
                    ],
                }],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {"label": {"type": "STRING"}},
                        "required": ["label"],
                    },
                    "thinkingConfig": {"thinkingBudget": 1},
                },
            }
        }))
    req_path = work / "requests.jsonl"
    req_path.write_text("\n".join(lines) + "\n")
    (work / "key_meta.json").write_text(json.dumps(key_meta))
    (work / "order.json").write_text(json.dumps(
        [k for k, _ in sorted(key_meta.items(), key=lambda kv: kv[1]["__index"])]
    ))
    (work / "gcs_job.txt").write_text(gcs_job)

    log(f"uploading {len(items)} flacs + jsonl -> {gcs_job}")
    try:
        _gcs_cp_recursive(str(flac_dir), f"{gcs_job}/flac")
        _gcs_cp(str(req_path), f"{gcs_job}/requests.jsonl")
    finally:
        shutil.rmtree(flac_dir, ignore_errors=True)
        for item in items:
            item.pop("flac", None)

    job = cli.batches.create(
        model=model,
        src=f"{gcs_job}/requests.jsonl",
        config={
            "display_name": f"t2a-{model}-{stamp}",
            "dest": f"{gcs_job}/out/",
        },
    )
    name = job.name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with JOBS_FILE.open("a") as jf:
        jf.write(json.dumps({
            "job": name,
            "model": model,
            "n": len(items),
            "work": str(work),
            "gcs": gcs_job,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state": "submitted",
            "mode": "gcs",
        }) + "\n")
    log(f"submitted {name} n={len(items)} model={model}")
    return name


def submit_with_fallback(
    cli, items: list[dict], work: Path, models: list[str] | None = None,
) -> tuple[str, str]:
    last: Exception | None = None
    for model in (models or FALLBACK_MODELS):
        try:
            return submit_batch(cli, model, items, work), model
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc)
            exhausted = any(
                s in msg.upper()
                for s in ("RESOURCE_EXHAUSTED", "QUOTA", "429", "UNAVAILABLE")
            )
            missing = "NOT_FOUND" in msg or "404" in msg
            log(f"{model} submit failed ({type(exc).__name__}): {msg[:240]}")
            if not (exhausted or missing) and model != FALLBACK_MODELS[-1]:
                # still try fallbacks for ACL/quota surprises
                pass
            continue
    raise RuntimeError(f"all models failed: {last}") from last


def _response_from_gcs_line(obj: dict):
    if "response" in obj:
        return obj["response"]
    if "candidates" in obj:
        return obj
    inner = obj.get("prediction") or obj.get("instance")
    if isinstance(inner, dict):
        return inner.get("response") or inner
    return obj


def harvest_job(cli, job_name: str, work: Path, model: str, out_f, train_f) -> str:
    job = cli.batches.get(name=job_name)
    state = str(getattr(job, "state", None) or "")
    state_s = state.split(".")[-1] if state else "UNKNOWN"
    if state_s not in {
        "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
    }:
        return state_s

    key_meta = json.loads((work / "key_meta.json").read_text())
    order = json.loads((work / "order.json").read_text()) if (work / "order.json").exists() else []

    if state_s != "JOB_STATE_SUCCEEDED":
        log(f"{job_name} finished {state_s} err={getattr(job, 'error', None)}")
        for meta in key_meta.values():
            row = {k: v for k, v in meta.items() if k not in ("__index", "gcs_flac")}
            write_row(out_f, train_f, {
                **row, "gemini_label": None, "error": f"batch {state_s}",
                "skip_reason": None, "labeler": f"batch:{model}",
            }, model)
        return state_s

    dest = getattr(job, "dest", None)
    gcs_uri = getattr(dest, "gcs_uri", None) if dest is not None else None
    if not gcs_uri:
        gcs_uri = (work / "gcs_job.txt").read_text().strip() + "/out/"
    local_out = work / "out"
    local_out.mkdir(exist_ok=True)
    listed = _gcs_ls(gcs_uri.rstrip("/") + "/**") or _gcs_ls(gcs_uri)
    jsonls = [u for u in listed if u.endswith(".jsonl") or "prediction" in u]
    responses: list = []
    if jsonls:
        for uri in jsonls:
            dest_file = local_out / Path(uri.rstrip("/").split("/")[-1])
            try:
                _gcs_cp(uri, str(dest_file))
            except subprocess.CalledProcessError:
                continue
            for line in dest_file.read_text().splitlines():
                if line.strip():
                    responses.append(json.loads(line))
    else:
        # copy whole prefix
        try:
            _gcs_cp_recursive(gcs_uri, str(local_out))
            for p in local_out.rglob("*"):
                if p.is_file() and p.suffix in {".jsonl", ".json"}:
                    for line in p.read_text().splitlines():
                        if line.strip():
                            try:
                                responses.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
        except subprocess.CalledProcessError as exc:
            log(f"gcs harvest copy failed: {exc}")

    ok = err = 0
    for idx, item in enumerate(responses):
        key = order[idx] if idx < len(order) else None
        meta = key_meta.get(key) if key else None
        if not meta:
            continue
        meta = {k: v for k, v in meta.items() if k not in ("__index", "gcs_flac")}
        if isinstance(item, dict) and item.get("error") and not item.get("response"):
            write_row(out_f, train_f, {
                **meta, "gemini_label": None, "error": str(item.get("error"))[:300],
                "skip_reason": None, "labeler": f"batch:{model}",
            }, model)
            err += 1
            continue
        resp = _response_from_gcs_line(item) if isinstance(item, dict) else item
        label, error = parse_label_from_response(resp)
        row = {
            **meta,
            "gemini_label": label,
            "error": error,
            "skip_reason": None,
            "labeler": f"batch:{model}",
        }
        if label and label != "reject":
            row.update(to_clap_row(row, label, labeler=f"batch:{model}"))
        write_row(out_f, train_f, row, model)
        if error:
            err += 1
        else:
            ok += 1
    log(f"harvested {job_name}: ok={ok} err={err} n_resp={len(responses)}")
    shutil.rmtree(work / "flac", ignore_errors=True)
    shutil.rmtree(work / "out", ignore_errors=True)
    return state_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="Override fallback chain with a single model")
    ap.add_argument(
        "--stripe-models",
        action="store_true",
        help="Round-robin Vertex batch submits across GEMINI_BATCH_MODELS (3.8/3.7/3.6 pools)",
    )
    ap.add_argument("--batch-size", type=int, default=80)
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--until-empty", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--file-list", type=Path, default=HERE / "label_tool" / "audios2_file_list.json")
    ap.add_argument("--poll", action="store_true")
    ap.add_argument("--poll-only", action="store_true")
    ap.add_argument("--poll-sec", type=int, default=45)
    ap.add_argument(
        "--max-inflight",
        type=int,
        default=int(os.environ.get("GEMINI_MAX_INFLIGHT", "80")),
        help="Vertex jobs to keep in flight before collect back-pressures",
    )
    ap.add_argument(
        "--collect-workers",
        type=int,
        default=max(16, (os.cpu_count() or 8) * 2),
        help="Threads for Hub+ffmpeg candidate prep",
    )
    args = ap.parse_args()
    global FALLBACK_MODELS, JOBS_FILE
    if args.model:
        FALLBACK_MODELS = [args.model]

    cli = client()
    model_tag = "batch.vertex"
    if args.num_shards > 1:
        model_tag = f"batch.vertex.s{args.shard_index:02d}"
        JOBS_FILE = OUT_DIR / f"gemini_batch_jobs.s{args.shard_index:02d}.jsonl"
    out_path = OUT_DIR / f"gemini_audios2.{model_tag}.jsonl"
    train_path = OUT_DIR / f"clap_train_audios2.{model_tag}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.poll_only:
        if not JOBS_FILE.exists():
            log("no jobs file")
            return 0
        with out_path.open("a") as out_f, train_path.open("a") as train_f:
            for line in JOBS_FILE.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("harvested"):
                    continue
                work = Path(rec["work"])
                if not work.exists():
                    continue
                state = harvest_job(cli, rec["job"], work, rec["model"], out_f, train_f)
                if state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                    rec["harvested"] = True
                    rec["state"] = state
                    with JOBS_FILE.open("a") as jf:
                        jf.write(json.dumps(rec) + "\n")
        return 0

    round_i = 0
    while True:
        round_i += 1
        done = load_done(ledger_paths())
        log(
            f"round={round_i} done_ledger={len(done)} models={FALLBACK_MODELS} "
            f"workers={args.collect_workers} shard={args.shard_index}/{args.num_shards} "
            f"max_inflight={args.max_inflight}"
        )
        files = json.loads(args.file_list.read_text())
        stage = stage_dir_for(args.shard_index, args.num_shards)
        stage.mkdir(parents=True, exist_ok=True)

        if args.until_empty and not args.limit:
            limit = 0
        else:
            limit = args.limit or (args.batch_size * max(args.max_batches, 1))
        n_batches = 0
        n_api = 0
        n_rej = 0
        pending: dict[str, tuple[Path, str]] = {}
        terminal = {
            "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
        }

        with out_path.open("a") as out_f, train_path.open("a") as train_f:
            def harvest_ready(*, noisy: bool = False) -> None:
                done_jobs = []
                for name, (work, model) in list(pending.items()):
                    state = harvest_job(cli, name, work, model, out_f, train_f)
                    if noisy or state in terminal:
                        log(f"poll {name} -> {state}")
                    if state in terminal:
                        rec = {
                            "job": name, "model": model, "work": str(work),
                            "harvested": True, "state": state,
                        }
                        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
                        with JOBS_FILE.open("a") as jf:
                            jf.write(json.dumps(rec) + "\n")
                        done_jobs.append(name)
                for name in done_jobs:
                    pending.pop(name, None)

            def wait_for_slot() -> None:
                while len(pending) >= max(1, args.max_inflight):
                    harvest_ready(noisy=False)
                    if len(pending) >= max(1, args.max_inflight):
                        time.sleep(args.poll_sec)

            def on_local_reject(rows: list[dict]) -> None:
                nonlocal n_rej
                for row in rows:
                    r = {k: v for k, v in row.items() if not k.startswith("_") and k not in ("flac", "prompt")}
                    write_row(out_f, train_f, r, "local-silence")
                    n_rej += 1

            stripe_i = 0

            def on_api_batch(chunk: list[dict]) -> None:
                nonlocal n_batches, n_api, stripe_i
                if not chunk:
                    return
                if not args.until_empty and n_batches >= args.max_batches:
                    return
                wait_for_slot()
                work = OUT_DIR / "batch_work" / f"vertex_{int(time.time())}_{n_batches}_{os.getpid()}"
                chain = FALLBACK_MODELS
                if args.stripe_models and len(FALLBACK_MODELS) > 1:
                    i = stripe_i % len(FALLBACK_MODELS)
                    stripe_i += 1
                    chain = FALLBACK_MODELS[i:] + FALLBACK_MODELS[:i]
                name, model = submit_with_fallback(cli, chunk, work, models=chain)
                pending[name] = (work, model)
                n_batches += 1
                n_api += len(chunk)
                harvest_ready(noisy=False)
                if n_batches % 10 == 0:
                    log(f"in_flight={len(pending)} submitted_batches={n_batches} api={n_api}")

            leftover = collect_candidates(
                files=files, done=done, num_shards=args.num_shards,
                shard_index=args.shard_index, limit=limit, stage=stage,
                workers=args.collect_workers,
                on_api_batch=on_api_batch,
                api_batch_size=args.batch_size,
                on_local_reject=on_local_reject,
            )
            if leftover:
                on_local_reject([x for x in leftover if x.get("_local_reject")])
                rest = [x for x in leftover if not x.get("_local_reject")]
                if rest:
                    on_api_batch(rest)

            log(f"collected api={n_api} local_reject={n_rej} in_flight={len(pending)}")
            if not n_api and not n_rej and not pending:
                log("no remaining candidates")
                break

            if not args.poll:
                log("submitted; re-run with --poll-only later to harvest")
                return 0

            while pending:
                harvest_ready(noisy=True)
                if pending:
                    time.sleep(args.poll_sec)

        if not args.until_empty:
            break
        if not n_api:
            break
    log("batch annotate done")
    # A round with zero API work and zero local rejects is usually auth/Hub
    # failure, not an empty backlog — do not claim completion.
    if args.until_empty and (n_api or n_rej):
        marker = OUT_DIR / "GEMINI_ALL_DONE"
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        log(f"wrote {marker}")
    elif args.until_empty:
        log("not marking GEMINI_ALL_DONE (empty round, likely a fetch failure)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
