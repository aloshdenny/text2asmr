"""ASMR trigger ontology and the CLAP prompts used to detect each trigger.

The released dataset ships no trigger annotations (see docs/DATA_NOTES.md), so
the tags the T2A paper describes have to be reconstructed from the audio.  We
do that with zero-shot audio-text matching: each trigger carries a set of
natural-language probes, and a segment is tagged with whichever trigger scores
highest -- provided it clears a margin over the ``NEGATIVE`` probes, which exist
to catch segments that are really speech, silence, or room tone.

The four triggers the paper names are marked ``in_paper``; the rest extend the
ontology to what ASMR audio actually contains.  Keep ``key`` values stable, as
they become the tag vocabulary in the published dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Trigger:
    key: str
    probes: tuple[str, ...]
    in_paper: bool = False
    #: Triggers whose character is defined by transients (taps, clicks) need a
    #: shorter analysis window than sustained ones (brushing, whispering).
    transient: bool = False

    @property
    def caption(self) -> str:
        """Caption written into the audio-model training set."""
        return f"ASMR {self.key}, close-mic binaural recording, no speech"


TRIGGERS: tuple[Trigger, ...] = (
    Trigger("brushing", (
        "brushing sounds", "a soft brush moving across a microphone",
        "bristles sweeping",
    ), in_paper=True),
    Trigger("tapping", (
        "tapping on a hard surface", "fingernails tapping wood",
        "rhythmic tapping",
    ), in_paper=True, transient=True),
    Trigger("crinkling", (
        "crinkling plastic", "crumpling a wrapper", "crinkling packaging",
    ), in_paper=True),
    Trigger("rustling", (
        "rustling fabric", "rustling paper", "soft rustling",
    ), in_paper=True),
    Trigger("scratching", (
        "scratching a textured surface", "fingernails scratching",
    ), transient=True),
    Trigger("clinking", (
        "small objects clinking together", "jewelry or utensils clinking",
        "delicate metallic clinking",
    ), transient=True),
    Trigger("breathing close", (
        "close-mic breathing directly into the microphone",
        "soft breathy exhale right next to the mic",
    )),
    Trigger("breathing heavy", (
        "heavy deep breathing", "labored breathing", "breathing with exertion",
    )),
    Trigger("kissing", (
        "kissing sounds", "soft kiss sounds close to a microphone",
    ), transient=True),
    Trigger("footsteps", (
        "soft footsteps", "footsteps on a floor",
    ), transient=True),
    Trigger("fabric rustling", (
        "rustling clothing fabric", "fabric rubbing together",
    )),
    Trigger("paper rustling", (
        "rustling paper", "crinkling a paper bag",
    )),
    Trigger("page turning", (
        "turning pages of a book", "paper pages flipping",
    ), transient=True),
    Trigger("liquid", (
        "pouring liquid", "water sloshing in a bottle", "stirring a drink",
    )),
    Trigger("mouth sounds", (
        "soft mouth sounds", "lip smacking close to a microphone",
    )),
    Trigger("breathing", (
        "slow deep breathing", "soft breath close to a microphone",
    )),
    Trigger("blowing", (
        "blowing air gently into a microphone", "soft breath blowing",
    )),
    Trigger("hand movements", (
        "hands rubbing together", "hand movements near a microphone",
    )),
    Trigger("microphone touching", (
        "touching a fuzzy microphone cover", "rubbing a microphone windscreen",
    )),
    Trigger("glass", (
        "tapping a glass jar", "glass objects clinking",
    ), transient=True),
    Trigger("wood", (
        "tapping a wooden box", "wooden objects knocking",
    ), transient=True),
    Trigger("sticky", (
        "sticky slime sounds", "sticky tape peeling",
    )),
    Trigger("cutting", (
        "scissors cutting hair", "snipping scissors",
    ), transient=True),
)

#: Probes for content that must NOT be tagged as a trigger.  A candidate
#: segment is rejected unless its best trigger score beats the best negative
#: score by ``MARGIN``.
NEGATIVE: tuple[str, ...] = (
    "a person speaking",
    "a person whispering words",
    "silence",
    "quiet room tone",
    "background hum",
    "music playing",
)

MARGIN: float = 0.05

BY_KEY: dict[str, Trigger] = {t.key: t for t in TRIGGERS}
PAPER_TRIGGERS: tuple[str, ...] = tuple(t.key for t in TRIGGERS if t.in_paper)


def all_probes() -> tuple[list[str], list[str | None]]:
    """Flatten every probe into a list, plus the trigger key each maps to.

    Negative probes map to ``None``.  Returning them together lets the tagger
    embed all text once and score positives and negatives in a single pass.
    """
    texts: list[str] = []
    owners: list[str | None] = []
    for trigger in TRIGGERS:
        for probe in trigger.probes:
            texts.append(probe)
            owners.append(trigger.key)
    for probe in NEGATIVE:
        texts.append(probe)
        owners.append(None)
    return texts, owners


def spatial_descriptor(stereo, threshold_db: float = 2.5) -> str:
    """Classify a stereo clip's channel balance as a caption descriptor.

    Pure interaural-level-difference DSP, no model: split the clip in half,
    compare left/right RMS in each half. A clip whose balance sits on one side
    throughout is "panned left/right"; one whose balance flips between halves
    is "moving" -- captioning that distinction is what lets a diffusion model
    learn actual spatial movement instead of a static, centered stereo image,
    which is what "surreal surround" is really asking for.

    ``stereo`` is (2, samples) float audio, matching what the builder decodes.
    """
    import numpy as np

    if stereo.shape[0] != 2 or stereo.shape[1] < 4:
        return "centered"

    def balance_db(chunk) -> float:
        l = float(np.sqrt(np.mean(np.square(chunk[0], dtype=np.float64))))
        r = float(np.sqrt(np.mean(np.square(chunk[1], dtype=np.float64))))
        return 20.0 * (np.log10(max(l, 1e-9)) - np.log10(max(r, 1e-9)))

    mid = stereo.shape[1] // 2
    b1, b2 = balance_db(stereo[:, :mid]), balance_db(stereo[:, mid:])

    def side(b: float) -> str:
        if b > threshold_db:
            return "left"
        if b < -threshold_db:
            return "right"
        return "center"

    s1, s2 = side(b1), side(b2)
    if s1 != s2 and {"left", "right"} <= {s1, s2}:
        return "moving across"
    if s1 == s2 == "center":
        return "centered"
    return f"panned {s1 if s1 != 'center' else s2}"


def intensity_from_loudness(rms_db: float, floor: float = -45.0,
                            ceil: float = -12.0) -> tuple[str, float]:
    """Map a segment's loudness onto the paper's intensity vocabulary.

    The paper introduces ``[mild] [soft] [vigorous] [loud]`` as intensity
    prefixes but never defines them operationally.  Loudness relative to the
    corpus is the one proxy available from audio alone; it correlates with
    vigour for contact triggers (harder tapping is louder) though not
    perfectly for all of them.
    """
    span = max(ceil - floor, 1e-6)
    scaled = min(max((rms_db - floor) / span, 0.0), 1.0)
    if scaled < 0.25:
        return "mild", scaled
    if scaled < 0.5:
        return "soft", scaled
    if scaled < 0.78:
        return "vigorous", scaled
    return "loud", scaled
