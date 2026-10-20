"""Sounds a recording's title promises, as menu labels (the labels added 2026-10-18 that no machine judge knows yet).

A soundgasm uid carries its recording's title ("creator/F4M-...-Spanking-Wet-Sounds.m4a_000123450"). A 6-second window
from a recording that promises a sound may well not contain it, so these tags only decide what people hear first
(private.thin_boost); they are never used as labels. The patterns are the ones the tag counts behind the NSFW menu
came from (71,353 titled recordings in aoxo/t2a-mommy and aoxo/t2a-daddy).
"""
from __future__ import annotations
import re

TAGS = {
    "sex slapping": r"slap|clap|skin on skin|pounding|fap sound",
    "wet sounds": r"wet sound|squelch|wetsound|wet noise|fingering|fleshlight|\blube\b|lotion",
    "squirting": r"squirt",
    "spanking": r"spank",
    "vibrator": r"vibrat|hitachi|\bwand\b|thruster|\btoys?\b",
    "gagging": r"\bgag|deep ?throat",
    "whimpering": r"whimper",
    "growling": r"growl",
    "grunting": r"grunt|groan",
    "bed creaking": r"creak|squeak|bed sound",
    "laughing": r"laugh|giggl|chuckl",            # 2026-10-20, from labellers' free text
    "gasping": r"gasp",
}
_TAGS = {k: re.compile(p) for k, p in TAGS.items()}


def title_of(uid: str) -> str:
    """The title words of a soundgasm uid (empty for YouTube / Freesound uids and hash-named files)."""
    if ":" in uid.split("/")[0]: return ""
    stem = re.sub(r"\.(m4a|mp3)(_\d+)?$", "", uid).split("/")[-1]
    if re.fullmatch(r"[0-9a-f]{32,64}", stem): return ""
    return " " + re.sub(r"[-_.]+", " ", stem).lower() + " "


def title_tags(uid: str) -> list[str]:
    t = title_of(uid)
    return [k for k, p in _TAGS.items() if t and p.search(t)]
