#!/usr/bin/env python3
"""Re-tag every trigger clip using Gemini instead of CLAP's zero-shot scoring.

retag_triggers.py's CLAP-based tagging is what the labeling tool was built to
patch (its low-confidence misses), but the standing worry is broader: CLAP's
guesses on clips that DID clear its margin aren't verified either. This runs
every one of the corpus's 20,332 trigger clips (25.00 hours total) through
Gemini instead, as a full second opinion rather than a spot-check.

Each clip's source field points back to aoxo/audios (the pre-transcribed
corpus these clips were originally cut from), which already carries
word-level transcripts from an earlier transcription pass -- so each request
includes the 2 words immediately before and after the clip's time window as
text context, giving Gemini continuity the isolated audio alone doesn't have
(e.g. a creator saying "some tapping now" right before the clip is a much
stronger signal than the sound in isolation).

Resumable like everything else in this pipeline: results append to
gemini_retag.jsonl keyed by file_name, and a re-run only processes what
isn't already in there.

    python3 scripts/retag_triggers_gemini.py --limit 30   # smoke test
    python3 scripts/retag_triggers_gemini.py               # full run
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text2asmr.data.ontology import TRIGGERS

REPO_SEGMENTS = "aoxo/text2asmr-segments"
REPO_SOURCE = "aoxo/audios"  # pre-transcribed source corpus these clips were cut from

HERE = Path(__file__).resolve().parents[1]
OUT_FILE = HERE / "label_tool" / "gemini_retag.jsonl"
CONTEXT_WORDS = 2

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

# NeoCustoms Document AI GCP project — Vertex generateContent works with ADC
# (no AI Studio API key). AI Studio keys on the old account return 403.
VERTEX_PROJECT = os.environ.get("GEMINI_VERTEX_PROJECT", "project-9da5a2fe-3df4-485e-9a9")
VERTEX_LOCATION = os.environ.get("GEMINI_VERTEX_LOCATION", "global")
GEMINI_BACKEND = os.environ.get("GEMINI_BACKEND", "vertex").strip().lower()

_vertex_tok = {"v": "", "ts": 0.0, "lock": threading.Lock()}


def set_model(model: str) -> None:
    """Switch the module-level Gemini endpoint (for multi-model parallel runs)."""
    global MODEL, API_URL
    MODEL = model
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def _vertex_token() -> str:
    now = time.time()
    with _vertex_tok["lock"]:
        if _vertex_tok["v"] and now - _vertex_tok["ts"] < 1400:
            return _vertex_tok["v"]
        import google.auth
        from google.auth.transport.requests import Request

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(Request())
        if not creds.token:
            raise RuntimeError("empty Vertex ADC token")
        _vertex_tok["v"] = creds.token
        _vertex_tok["ts"] = now
        return creds.token


def _vertex_url(model: str) -> str:
    loc = VERTEX_LOCATION
    host = (
        "aiplatform.googleapis.com"
        if loc == "global"
        else f"{loc}-aiplatform.googleapis.com"
    )
    return (
        f"https://{host}/v1/projects/{VERTEX_PROJECT}/locations/{loc}"
        f"/publishers/google/models/{model}:generateContent"
    )


TRIGGER_KEYS = sorted(t.key for t in TRIGGERS)

_transcript_cache: dict[str, list[dict]] = {}
_cache_lock = None  # set in main() once we know we're threaded


def load_transcript(source: str) -> list[dict]:
    """Cached per source file -- many clips share the same source, and this
    avoids re-downloading the same transcript for each one."""
    if source in _transcript_cache:
        return _transcript_cache[source]
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(REPO_SOURCE, f"{source}.json", repo_type="dataset")
        entries = json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001 - a missing/renamed source file just means no context
        entries = []
    _transcript_cache[source] = entries
    return entries


def context_words(source: str, start: float, duration: float, n: int = CONTEXT_WORDS) -> tuple[str, str]:
    entries = load_transcript(source)
    words = [e for e in entries if e["type"] == "word"]
    before = [w["word"] for w in words if w["end"] <= start][-n:]
    after = [w["word"] for w in words if w["start"] >= start + duration][:n]
    return " ".join(before), " ".join(after)


def build_prompt(before: str, after: str) -> str:
    keys = ", ".join(TRIGGER_KEYS)
    context_line = ""
    if before or after:
        context_line = (
            f"\nSurrounding speech for context (what was said just before/after this clip "
            f"in the source recording -- may be unrelated or empty, use it only if it "
            f"actually helps): before=\"{before}\" after=\"{after}\"\n"
        )
    return (
        "You are classifying a short audio clip from an ASMR audio corpus. "
        "Listen to the clip and decide what it actually contains.\n\n"
        f"Known trigger sound categories: {keys}\n"
        f"{context_line}\n"
        "Rules, in priority order:\n"
        "1. If the clip reasonably fits one of the known categories -- including close or "
        "approximate matches, e.g. 'wood tapping' or 'finger tapping' both belong under "
        "'tapping', not a new label -- answer with that exact category name. Prefer an "
        "existing category over inventing a new one whenever the sound is fundamentally the "
        "same kind of thing, even if the material, intensity, or exact technique differs.\n"
        "2. If the clip is NOT a non-speech trigger sound at all (e.g. it's actually speech, "
        "moaning, silence, or something else that isn't a discrete sound effect), answer "
        "with exactly: reject\n"
        "3. Only if the clip is a real, distinct non-speech sound that genuinely isn't a "
        "variant of any known category (a different kind of sound entirely, not just a "
        "different flavor of an existing one) -- answer with a short (1-3 word), generic "
        "category name for it, not a specific description. Write it the way a new entry in "
        "the known list above would look (e.g. 'zipper' or 'page turning', not 'the sound of "
        "a metal zipper being pulled slowly'), so it can be reused as-is for other similar "
        "clips instead of every clip getting its own one-off phrasing.\n\n"
        "Respond with JSON: {\"label\": \"...\"}"
    )


SCHEMA = {"type": "OBJECT", "properties": {"label": {"type": "STRING"}}, "required": ["label"]}


_thread_local = threading.local()


def _session() -> requests.Session:
    """Per-thread Session with a large pool so high worker counts reuse sockets."""
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=64, pool_maxsize=64, max_retries=0
        )
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _thread_local.session = sess
    return sess


def _parse_label_payload(payload: dict, raw_text: str) -> tuple[str | None, str | None]:
    feedback = payload.get("promptFeedback") or {}
    if feedback.get("blockReason") == "PROHIBITED_CONTENT":
        return "reject", None
    cands = payload.get("candidates") or []
    if not cands:
        return None, f"no candidates: {raw_text[:300]}"
    finish = cands[0].get("finishReason")
    parts = ((cands[0].get("content") or {}).get("parts") or [])
    if finish == "SAFETY" or not parts:
        return "reject", None
    text = parts[0].get("text") or ""
    return json.loads(text)["label"].strip().lower(), None


def classify(api_key: str, audio_bytes: bytes, prompt: str,
             model: str | None = None) -> tuple[str | None, str | None]:
    """Returns (label, error). error is None on success, so callers can tell
    a genuine API failure apart from the label legitimately being absent --
    the first version of this returned None for both, which meant a whole
    smoke test failing on a billing error looked identical in the output to
    a parsing bug, with no way to tell which from gemini_retag.jsonl alone."""
    if model and model != MODEL:
        set_model(model)
    data = base64.b64encode(audio_bytes).decode()
    use_vertex = GEMINI_BACKEND in ("vertex", "vertex_ai", "gcp") or not (api_key or "").strip()
    if use_vertex:
        return _classify_vertex(data, prompt)
    return _classify_ai_studio(api_key, data, prompt)


def _classify_ai_studio(api_key: str, data: str, prompt: str) -> tuple[str | None, str | None]:
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "audio/flac", "data": data}},
            {"text": prompt},
        ]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "temperature": 0,
            # This is a closed classification call, not a reasoning task --
            # left at the default, the model spends ~270 hidden "thinking"
            # tokens per call (measured), which get billed at the output
            # rate and roughly quintuple the real cost over the visible
            # dozen-token JSON answer. 0 is rejected as invalid; 1 is the
            # lowest accepted value and empirically drops thoughtsTokenCount
            # to 0.
            "thinkingConfig": {"thinkingBudget": 1},
        },
    }
    last_err = None
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    for attempt in range(5):
        try:
            r = _session().post(f"{api_url}?key={api_key}", json=body, timeout=60)
        except requests.RequestException as exc:
            last_err = f"request failed: {exc}"
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429:
            last_err = f"429: {r.text[:300]}"
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        try:
            return _parse_label_payload(r.json(), r.text)
        except Exception as exc:  # noqa: BLE001 - a malformed response shouldn't kill the run
            return None, f"parse failed: {exc}: {r.text[:300]}"
    return None, last_err


def _classify_vertex(data: str, prompt: str) -> tuple[str | None, str | None]:
    """Vertex generateContent on the NeoCustoms GCP project (ADC, no API key)."""
    body_with_think = {
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "audio/flac", "data": data}},
            {"text": prompt},
        ]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "temperature": 0,
            "thinkingConfig": {"thinkingBudget": 1},
        },
    }
    body_plain = {
        "contents": body_with_think["contents"],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "temperature": 0,
        },
    }
    last_err = None
    url = _vertex_url(MODEL)
    body = body_with_think
    for attempt in range(5):
        try:
            headers = {
                "Authorization": f"Bearer {_vertex_token()}",
                "x-goog-user-project": VERTEX_PROJECT,
                "Content-Type": "application/json",
            }
            r = _session().post(url, json=body, headers=headers, timeout=60)
        except requests.RequestException as exc:
            last_err = f"request failed: {exc}"
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429:
            last_err = f"429: {r.text[:300]}"
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 400 and "thinking" in r.text.lower() and body is body_with_think:
            body = body_plain
            continue
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        try:
            return _parse_label_payload(r.json(), r.text)
        except Exception as exc:  # noqa: BLE001
            return None, f"parse failed: {exc}: {r.text[:300]}"
    return None, last_err


def process_one(api_key: str, row: dict) -> dict:
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(REPO_SEGMENTS, f"triggers/{row['file_name']}", repo_type="dataset")
        audio_bytes = Path(p).read_bytes()
    except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the run
        return {"file_name": row["file_name"], "gemini_label": None, "error": str(exc)}

    before, after = context_words(row.get("source", ""), row.get("start", 0), row.get("duration", 0))
    prompt = build_prompt(before, after)
    label, error = classify(api_key, audio_bytes, prompt)
    return {
        "file_name": row["file_name"],
        "gemini_label": label,
        "error": error,
        "clap_label": row.get("trigger"),
        "clap_confidence_v2": row.get("clap_confidence_v2"),
        "context_before": before,
        "context_after": after,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining clips")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set (source ~/.t2a_env first)")

    from huggingface_hub import hf_hub_download, list_repo_files
    meta_path = hf_hub_download(REPO_SEGMENTS, "triggers/metadata_v2.jsonl", repo_type="dataset")
    all_rows = [json.loads(l) for l in open(meta_path) if l.strip()]

    # metadata_v2.jsonl still lists clips that have since been pruned from the
    # repo -- filtering against the real file listing up front avoids burning
    # API calls (and Gemini credits) on 404s, same fix the labeling tool's
    # candidate pool needed for the same underlying reason.
    files = set(list_repo_files(REPO_SEGMENTS, repo_type="dataset"))
    real_names = {f.split("/")[-1] for f in files if f.startswith("triggers/") and f.endswith(".flac")}
    rows = [r for r in all_rows if r["file_name"] in real_names]
    print(f"{len(rows)} clips have real audio on the Hub (of {len(all_rows)} in metadata)", flush=True)

    done = set()
    if OUT_FILE.exists():
        # Only a real success counts as done -- a row with gemini_label None
        # (billing block, transient error, etc.) must not be permanently
        # skipped just because a row for it already exists on disk.
        done = {
            json.loads(l)["file_name"] for l in OUT_FILE.read_text().splitlines() if l.strip()
            and json.loads(l).get("gemini_label") is not None
        }
    todo = [r for r in rows if r["file_name"] not in done]
    print(f"{len(rows)} total clips, {len(done)} already done, {len(todo)} remaining", flush=True)
    if args.limit:
        todo = todo[: args.limit]
        print(f"limiting to {len(todo)} for this run", flush=True)

    n = 0
    errors = 0
    with OUT_FILE.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, api_key, r): r for r in todo}
        for fut in as_completed(futures):
            result = fut.result()
            out.write(json.dumps(result) + "\n")
            out.flush()
            n += 1
            if result.get("gemini_label") is None:
                errors += 1
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} done ({errors} errors so far)", flush=True)

    print(f"\n{n} clips processed ({errors} errors) -> {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
