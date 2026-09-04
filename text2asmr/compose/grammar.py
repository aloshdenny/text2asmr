"""Parser for the T2A bracket-tag script grammar.

The grammar comes from the T2A paper (NSCTC 2024), section IV.A steps 6-8:
transcripts interleave plain speech with bracketed trigger tags such as
``[brushing]``, optionally preceded by a bracketed intensity modifier such as
``[soft]``.  A transcript is therefore a *script*: an ordered sequence of
speech spans and trigger events that a renderer turns into audio.

The intensity vocabulary is closed (the paper enumerates it).  The trigger
vocabulary is deliberately open -- the paper says "brushing, rustling, tapping,
crinkling etc" -- so triggers are whatever bracket tokens are not intensities.
Call :func:`survey_vocabulary` over a corpus to recover the actual trigger set
rather than hardcoding one.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal

# Closed set, from the paper: "[mild], [soft], [vigorous] and [loud] could
# precede the aforementioned trigger sound tags".
INTENSITIES: dict[str, float] = {
    "mild": 0.35,
    "soft": 0.5,
    "vigorous": 0.85,
    "loud": 1.0,
}
DEFAULT_INTENSITY = 0.65

_TAG = re.compile(r"\[([^\[\]]+)\]")


@dataclass(frozen=True)
class Speech:
    """A span of spoken text, rendered by the TTS backend."""

    text: str
    kind: Literal["speech"] = "speech"


@dataclass(frozen=True)
class Trigger:
    """A non-speech ASMR trigger, rendered by the audio-diffusion backend."""

    name: str
    intensity: float = DEFAULT_INTENSITY
    modifier: str | None = None
    kind: Literal["trigger"] = "trigger"

    @property
    def prompt(self) -> str:
        """Natural-language prompt for a text-to-audio model.

        Bracket tags are corpus notation, not something a pretrained audio
        model has ever seen, so they are expanded into a caption in the register
        the base model was actually trained on.
        """
        lead = f"{self.modifier} " if self.modifier else ""
        return f"ASMR {lead}{self.name}, close-mic binaural, no speech"


Segment = Speech | Trigger


@dataclass
class Script:
    """An ordered, renderable interpretation of one transcript."""

    segments: list[Segment] = field(default_factory=list)
    source: str = ""

    @property
    def speech(self) -> list[Speech]:
        return [s for s in self.segments if isinstance(s, Speech)]

    @property
    def triggers(self) -> list[Trigger]:
        return [s for s in self.segments if isinstance(s, Trigger)]

    @property
    def is_pure_speech(self) -> bool:
        return bool(self.speech) and not self.triggers

    @property
    def is_pure_trigger(self) -> bool:
        return bool(self.triggers) and not self.speech

    def __len__(self) -> int:
        return len(self.segments)


def _clean(text: str) -> str:
    """Collapse whitespace left behind by tag removal."""
    return re.sub(r"\s+", " ", text).strip()


def parse(transcript: str) -> Script:
    """Parse one transcript into a :class:`Script`.

    An intensity tag binds to the trigger tag that follows it.  A dangling
    intensity -- one at end of string, or one followed by speech rather than a
    trigger -- is dropped, since it modifies nothing.
    """
    script = Script(source=transcript)
    pending: str | None = None
    cursor = 0

    for match in _TAG.finditer(transcript):
        between = _clean(transcript[cursor : match.start()])
        if between:
            # Speech intervened, so a pending intensity has nothing to bind to.
            pending = None
            script.segments.append(Speech(between))
        cursor = match.end()

        tag = match.group(1).strip().lower()
        if tag in INTENSITIES:
            pending = tag
            continue

        script.segments.append(
            Trigger(
                name=tag,
                intensity=INTENSITIES.get(pending, DEFAULT_INTENSITY),
                modifier=pending,
            )
        )
        pending = None

    tail = _clean(transcript[cursor:])
    if tail:
        script.segments.append(Speech(tail))

    return script


def survey_vocabulary(transcripts: Iterable[str]) -> tuple[Counter, Counter]:
    """Recover the corpus's actual trigger and intensity vocabularies.

    Returns ``(triggers, intensities)`` as frequency counters.  Use this to
    check what the dataset really contains before committing to a tag ontology.
    """
    triggers: Counter = Counter()
    intensities: Counter = Counter()
    for transcript in transcripts:
        for tag in _TAG.findall(transcript or ""):
            tag = tag.strip().lower()
            (intensities if tag in INTENSITIES else triggers)[tag] += 1
    return triggers, intensities


def iter_render_plan(script: Script) -> Iterator[tuple[int, Segment]]:
    """Yield ``(index, segment)`` in playback order."""
    yield from enumerate(script.segments)
