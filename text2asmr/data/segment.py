"""Turn word-level alignments into speech and trigger segments.

``aoxo/audios`` pairs each ``N.m4a`` with an ``N.json`` holding a flat list of
``{"type": "word"|"silence", "start", "end", ...}`` entries.  That alignment is
the whole basis of the split:

  * runs of ``word`` entries become **speech** segments for the TTS finetune
  * ``silence`` entries are only silent in the sense that the ASR found no
    words there.  In ASMR the gaps between whispered phrases are where the
    brushing and tapping live, so loud gaps become **trigger** candidates and
    quiet ones are discarded as true room tone.

Nothing here decodes audio; it works purely on the alignment so it can be
tested without media.  :func:`measure_loudness` is applied later, by the
extractor, once samples are available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

# Speech shorter than this is usually a stray word the aligner dropped out of a
# longer phrase; longer than this exceeds what the TTS backends train on.
MIN_SPEECH_S = 1.0
MAX_SPEECH_S = 20.0

# Trigger windows need to be long enough to carry texture but short enough that
# the audio model sees a single consistent event.
MIN_TRIGGER_S = 1.5
MAX_TRIGGER_S = 12.0

# A gap this long inside a phrase ends the phrase.  ASMR pacing is slow and
# deliberate, so this is looser than a conversational-speech default would be.
PHRASE_GAP_S = 0.7

# Most ASMR videos open with a few seconds to a minute of generic branded
# intro (music sting, channel jingle) before any speech starts. The ASR finds
# no words there, so the word-gap heuristic below treats it exactly like a
# real trigger gap -- with nothing to tell "intro music" apart from "brushing
# with no talking over it" except that it happens to sit at the very start of
# the file. Dropping trigger candidates before this cutoff is a cheap, direct
# fix for a real contamination source confirmed by listening to samples: 4-8%
# of the corpus's trigger clips start under 30-60s into their source file.
INTRO_SKIP_S = 30.0


@dataclass(frozen=True)
class Span:
    """A time range within one source file."""

    source: str
    start: float
    end: float
    kind: Literal["speech", "trigger_candidate"]
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def uid(self) -> str:
        return f"{self.source}_{int(self.start * 1000):09d}"


def load_alignment(path: str | Path) -> list[dict]:
    """Read one alignment JSON, tolerating the entries being wrapped."""
    data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    if isinstance(data, dict):
        for key in ("segments", "words", "alignment", "result"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            raise ValueError(f"{path}: no entry list found in object")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of entries")
    return data


def _valid(entry: dict) -> bool:
    """Reject entries missing timing or inverted, which do occur."""
    try:
        start, end = float(entry["start"]), float(entry["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return end > start


def split_alignment(entries: Sequence[dict], source: str,
                    intro_skip_s: float = INTRO_SKIP_S) -> list[Span]:
    """Split one file's alignment into speech phrases and trigger candidates.

    Consecutive words are merged into a phrase until either a gap longer than
    ``PHRASE_GAP_S`` or ``MAX_SPEECH_S`` of accumulated audio forces a break.
    Merging matters: individual words are too short to finetune on, and the
    natural unit for ASMR speech is the phrase.
    """
    entries = [e for e in entries if _valid(e)]
    entries = sorted(entries, key=lambda e: float(e["start"]))

    spans: list[Span] = []
    words: list[dict] = []

    def flush() -> None:
        if not words:
            return
        start, end = float(words[0]["start"]), float(words[-1]["end"])
        text = " ".join(str(w.get("word", "")).strip() for w in words).strip()
        text = " ".join(text.split())
        if text and MIN_SPEECH_S <= end - start <= MAX_SPEECH_S:
            spans.append(Span(source, start, end, "speech", text))
        words.clear()

    for entry in entries:
        if entry.get("type") == "word":
            if words:
                gap = float(entry["start"]) - float(words[-1]["end"])
                run = float(entry["end"]) - float(words[0]["start"])
                if gap > PHRASE_GAP_S or run > MAX_SPEECH_S:
                    flush()
            words.append(entry)
            continue

        start, end = float(entry["start"]), float(entry["end"])
        # The aligner emits a `silence` entry between *every* pair of words,
        # usually a few tens of milliseconds. Those are within-phrase pauses,
        # not breaks: flushing on them would make every phrase one word long.
        # Only a gap beyond PHRASE_GAP_S actually ends a phrase.
        if end - start <= PHRASE_GAP_S:
            continue
        flush()
        if end - start < MIN_TRIGGER_S:
            continue
        if start < intro_skip_s:
            continue
        # Long gaps are chopped into windows rather than dropped; a 60 s pause
        # full of brushing is many training examples, not one unusable one.
        cursor = start
        while end - cursor >= MIN_TRIGGER_S:
            stop = min(cursor + MAX_TRIGGER_S, end)
            spans.append(Span(source, cursor, stop, "trigger_candidate"))
            cursor = stop

    flush()
    return sorted(spans, key=lambda s: s.start)


def summarise(spans: Iterable[Span]) -> dict:
    """Aggregate counts and durations, for the corpus report."""
    speech = [s for s in spans if s.kind == "speech"]
    triggers = [s for s in spans if s.kind == "trigger_candidate"]
    speech_s = sum(s.duration for s in speech)
    trigger_s = sum(s.duration for s in triggers)
    # Hours are for the corpus total; seconds keep per-file summaries legible,
    # where hours would round to zero.
    return {
        "speech_segments": len(speech),
        "speech_seconds": round(speech_s, 2),
        "speech_hours": round(speech_s / 3600, 2),
        "trigger_candidates": len(triggers),
        "trigger_seconds": round(trigger_s, 2),
        "trigger_hours": round(trigger_s / 3600, 2),
        "words": sum(len(s.text.split()) for s in speech),
    }
