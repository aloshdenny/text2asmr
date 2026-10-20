"""Which voice a soundgasm clip is: its recording's corpus, aoxo/t2a-daddy (male) or aoxo/t2a-mommy (female).

The site serves a share of male-voice clips first (setting male_share), so every clip pushed from those corpora carries
info.voice. Trigger recordings (YouTube, Freesound) have no voice.
"""
from __future__ import annotations
import re
from functools import lru_cache

CORPORA = {"aoxo/t2a-daddy": "male", "aoxo/t2a-mommy": "female"}


@lru_cache(maxsize=1)
def corpus_voices() -> dict[str, str]:
    """recording path ("creator/file.m4a") -> voice, from the corpora's file listings (two API calls, no downloads)."""
    from huggingface_hub import HfApi
    api = HfApi()
    return {f: v for repo, v in CORPORA.items() for f in api.list_repo_files(repo, repo_type="dataset")
            if f.endswith((".m4a", ".mp3"))}


def voice_of(uid: str) -> str | None:
    if ":" in uid.split("/")[0]: return None              # yt:..., fs:..., pool:... are not corpus windows
    return corpus_voices().get(re.sub(r"_\d+$", "", uid))
