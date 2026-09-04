"""Caption lookup for stable-audio-tools training on the ASMR trigger set.

`stable-audio-tools` calls ``get_custom_metadata(info, audio)`` per example and
merges the result into what the model sees. For an ``audio_dir`` dataset the
only identifying field is ``relpath``, so captions are joined from the
builder's ``metadata.jsonl`` on file name.

The caption carries both the trigger and its intensity ("ASMR soft brushing,
close-mic binaural, no speech") because the intensity prefix is half the
paper's conditioning vocabulary -- dropping it would make `[mild]` and
`[vigorous]` indistinguishable to the model.

Point the dataset config's ``custom_metadata_module`` at this file. The
metadata path is taken from TEXT2ASMR_TRIGGER_METADATA if set, otherwise assumed to
sit beside the audio directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_CAPTIONS: dict[str, dict] | None = None

# Fallback when a file has no metadata row. Better than an empty prompt, which
# would train the model to associate silence-of-text with real audio.
_FALLBACK = "ASMR trigger sound, close-mic binaural, no speech"


def _metadata_path() -> Path:
    env = os.environ.get("TEXT2ASMR_TRIGGER_METADATA")
    if env:
        return Path(env)
    audio_dir = os.environ.get("TEXT2ASMR_TRIGGER_DIR", "/workspace/out/triggers")
    return Path(audio_dir) / "metadata.jsonl"


def _load() -> dict[str, dict]:
    global _CAPTIONS
    if _CAPTIONS is not None:
        return _CAPTIONS
    path = _metadata_path()
    table: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = row.get("file_name")
            if name:
                # Key on the bare name: relpath may carry a shard subdirectory
                # depending on how the archive was unpacked.
                table[Path(name).name] = row
    _CAPTIONS = table
    return table


def get_custom_metadata(info, audio):  # noqa: ANN001 - signature fixed by caller
    table = _load()
    relpath = info.get("relpath") or info.get("path") or ""
    row = table.get(Path(relpath).name)
    if row is None:
        return {"prompt": _FALLBACK}
    return {
        "prompt": row.get("caption") or _FALLBACK,
        # Kept for inspection and for any later filtering by class or level.
        "trigger": row.get("trigger", ""),
        "intensity": row.get("intensity", ""),
    }
