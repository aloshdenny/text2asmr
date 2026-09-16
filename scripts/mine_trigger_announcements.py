#!/usr/bin/env python3
"""Mine trigger-sound candidates from transcripts, incrementally.

transcribe_audios2.py is still working through aoxo/audios2 (thousands of
files remain), so this can't assume the corpus is fully transcribed -- it has
to be safe to run today against whatever's transcribed so far, and safe to
run again next week against whatever's transcribed by then, without redoing
work or waiting for the whole corpus to finish first.

That falls out of two design choices rather than needing special-casing:
  - Eligibility is just "does a .json exist for this source file on the Hub
    right now" -- a file with no transcript yet is invisible to this run and
    simply becomes visible on a later one once transcribe_audios2.py reaches
    it. No separate "pending" tracking needed.
  - MINED_FILE records which source files' transcripts have already been
    scanned (independent of whether they yielded any candidates), so a
    re-run only pays for the delta -- newly-transcribed files since last
    time -- mirroring the done.txt pattern used elsewhere in this project.

The actual mining idea: creators narrate their triggers ("let's do some
tapping now") right before doing them, and transcribe_audios2.py's own
word/silence entry schema already demarcates exactly the non-speech stretch
that follows a word -- so a trigger keyword immediately followed by a
"silence" entry is a strong, free (no audio decoding, no model inference)
signal that that silence IS the trigger sound. This is the same idea
discussed for expanding the ontology without human labeling, applied
directly to the transcripts already being generated for another purpose.

Output feeds the same labeling tool built for CLAP's low-confidence clips
(text2asmr/label_tool/): new candidate clips land in its clips/ dir and get
appended to its candidates.jsonl with a `source_method` field distinguishing
them, so the one UI handles both without any changes.

    python3 scripts/mine_trigger_announcements.py
    python3 scripts/mine_trigger_announcements.py --limit 200  # smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ID = "aoxo/audios2"
LABEL_TOOL_DIR = Path(__file__).resolve().parents[1] / "label_tool"
MINED_FILE = LABEL_TOOL_DIR / "mined_sources.txt"
CANDIDATES_FILE = LABEL_TOOL_DIR / "candidates.jsonl"
CLIPS_DIR = LABEL_TOOL_DIR / "clips"
STAGE_DIR = Path(__file__).resolve().parents[1] / "mine_stage"

MAX_WINDOW_S = 8.0   # cap on how much of a silence gap counts as "the trigger", not dead air
MIN_WINDOW_S = 0.6   # shorter than this is probably just word-to-word breathing room, not a sound

# Seed keyword -> ontology trigger key map. Deliberately not exhaustive --
# this is a precision-over-recall first pass (a human confirms every
# candidate anyway), and "other" labels coming back from the labeling tool
# are the intended way to grow this list, not hand-tuning it upfront.
KEYWORDS: dict[str, str] = {
    "tap": "tapping", "taps": "tapping", "tapping": "tapping", "knock": "tapping",
    "brush": "brushing", "brushing": "brushing", "bristles": "brushing",
    "crinkle": "crinkling", "crinkling": "crinkling", "crinkly": "crinkling",
    "crumple": "crinkling", "crumpling": "crinkling", "wrapper": "crinkling",
    "rustle": "rustling", "rustling": "rustling", "rustly": "rustling",
    "scratch": "scratching", "scratching": "scratching", "scratchy": "scratching",
    "clink": "clinking", "clinking": "clinking", "clinky": "clinking",
    "breathe": "breathing", "breathing": "breathing", "breath": "breathing",
    "kiss": "kissing", "kisses": "kissing", "kissing": "kissing",
    "footstep": "footsteps", "footsteps": "footsteps",
    "fabric": "fabric rustling", "clothing": "fabric rustling",
    "page": "page turning", "pages": "page turning",
    "pour": "liquid", "pouring": "liquid", "sip": "liquid", "sipping": "liquid",
    "blow": "blowing", "blowing": "blowing",
    "scissors": "cutting", "snip": "cutting", "snipping": "cutting", "cutting": "cutting",
    "glass": "glass", "jar": "glass",
    "wood": "wood", "wooden": "wood", "box": "wood",
    "sticky": "sticky", "slime": "sticky", "tape": "sticky",
    "microphone": "microphone touching", "mic": "microphone touching",
}
WORD_RE = re.compile(r"[^a-z']+")


def eligible_sources(done_transcripts: set[str], mined: set[str]) -> list[str]:
    """Transcribed-but-unmined source files -- the actual delta for this run."""
    return sorted(
        t[: -len(".json")] for t in done_transcripts
        if t.endswith(".json") and t[: -len(".json")] not in mined
    )


def find_candidates(entries: list[dict]) -> list[tuple[float, float, str, str]]:
    """(start, end, trigger_key, matched_word) for each announced-then-silent trigger."""
    out = []
    for i, e in enumerate(entries):
        if e["type"] != "word":
            continue
        word = WORD_RE.sub("", e["word"].lower())
        trigger_key = KEYWORDS.get(word)
        if not trigger_key:
            continue
        # The silence immediately following this word, if any -- skip past
        # other words in between (e.g. "...some tapping for you", still
        # counts) but only look one step ahead, not an unbounded scan.
        nxt = entries[i + 1] if i + 1 < len(entries) else None
        if not nxt or nxt["type"] != "silence":
            continue
        start, end = nxt["start"], min(nxt["end"], nxt["start"] + MAX_WINDOW_S)
        if end - start < MIN_WINDOW_S:
            continue
        out.append((start, end, trigger_key, e["word"]))
    return out


def extract_clip(source_audio: Path, start: float, end: float, out_path: Path) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", str(source_audio), "-c:a", "flac", str(out_path)],
        capture_output=True, timeout=60,
    )
    return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all eligible source files")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    LABEL_TOOL_DIR.mkdir(exist_ok=True)
    CLIPS_DIR.mkdir(exist_ok=True)
    STAGE_DIR.mkdir(exist_ok=True)

    api = HfApi()
    all_files = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
    transcripts = {f for f in all_files if f.endswith(".json")}
    mined = set(l.strip() for l in MINED_FILE.read_text().splitlines() if l.strip()) if MINED_FILE.exists() else set()

    todo = eligible_sources(transcripts, mined)
    total_transcribed = len(transcripts)
    print(f"{total_transcribed} files transcribed so far, {len(mined)} already mined, "
        f"{len(todo)} newly eligible this run "
        f"(untranscribed files aren't visible yet -- re-run later to pick them up)")
    if args.limit:
        todo = todo[: args.limit]

    existing_candidates = (
        [json.loads(l) for l in CANDIDATES_FILE.read_text().splitlines() if l.strip()]
        if CANDIDATES_FILE.exists() else []
    )
    existing_names = {r["file_name"] for r in existing_candidates}
    new_candidates: list[dict] = []

    for i, source in enumerate(todo, 1):
        try:
            transcript_path = hf_hub_download(REPO_ID, source + ".json", repo_type="dataset")
            entries = json.loads(Path(transcript_path).read_text())
        except Exception as exc:  # noqa: BLE001 - one bad transcript must not kill the run
            print(f"  [{i}/{len(todo)}] {source}: transcript read failed: {exc}")
            continue

        hits = find_candidates(entries)
        if hits:
            try:
                audio_path = Path(hf_hub_download(REPO_ID, source, repo_type="dataset",
                                                local_dir=STAGE_DIR))
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(todo)}] {source}: audio download failed: {exc}")
                mined.add(source)
                continue

            stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(source).stem)[:60]
            for j, (start, end, trigger_key, word) in enumerate(hits):
                clip_name = f"mined_{stem}_{j}.flac"
                if clip_name in existing_names:
                    continue
                out_path = CLIPS_DIR / clip_name
                if extract_clip(audio_path, start, end, out_path):
                    new_candidates.append({
                        "file_name": clip_name,
                        "trigger": trigger_key,
                        "source": source,
                        "start": round(start, 3),
                        "duration": round(end - start, 3),
                        "matched_word": word,
                        "source_method": "transcript_mining",
                    })
            audio_path.unlink(missing_ok=True)

        mined.add(source)
        if i % 50 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {len(new_candidates)} candidates found so far", flush=True)
            MINED_FILE.write_text("\n".join(sorted(mined)) + "\n")

    MINED_FILE.write_text("\n".join(sorted(mined)) + "\n")

    if new_candidates:
        with CANDIDATES_FILE.open("a") as f:
            for r in new_candidates:
                f.write(json.dumps(r) + "\n")

    print(f"\n{len(new_candidates)} new trigger candidates mined from transcripts "
        f"-> {CANDIDATES_FILE}")
    print(f"{len(mined)}/{total_transcribed} transcribed files mined overall; "
        f"{total_transcribed - len(mined)} left in the currently-transcribed set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
