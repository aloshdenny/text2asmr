"""Prove the output path works *before* spending money or GPU hours on it, and never buffer results.

This exists because a judge run completed 450 paid Gemini calls and then failed to save them: the output
filename was built from a model id containing a slash, so the write went to a directory that did not exist.
The work was correct, the spend was real, and the data was gone.

Three rules, enforced here so every script gets them:

  1. ``preflight`` -- touch the real output path (write, fsync, read back, delete) before the expensive loop
     starts.  A path that cannot be written must fail in the first second, not the last.
  2. ``JsonlSink`` -- append and flush every record as it is produced.  A crash then costs one record, not
     a run.  ``fsync_every`` forces the OS to disk periodically for work that is expensive to repeat.
  3. ``safe_name`` -- model ids, repo ids and labels contain ``/`` and ``:``; they never go into a path raw.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable


def safe_name(s: str) -> str:
    """Filename-safe form of an id like 'or:google/gemini-3.1-pro-preview' or 'mouth sounds'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def preflight(*paths: str | Path, note: str = "") -> None:
    """Fail now if any of these paths cannot actually be written.

    Raises OSError with the offending path, so the message points at the fix rather than at a stack of
    library frames thrown 40 minutes into a paid run.
    """
    for p in paths:
        p = Path(p)
        target = p if p.suffix else p / ".preflight"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            probe = target.with_name(target.name + ".preflight")
            payload = f"preflight {time.time()}\n"
            with open(probe, "w") as fh:
                fh.write(payload); fh.flush(); os.fsync(fh.fileno())
            if probe.read_text() != payload:
                raise OSError(f"read-back mismatch at {probe}")
            probe.unlink()
        except Exception as e:
            raise OSError(f"output path is not writable: {p} ({type(e).__name__}: {e}){' -- ' + note if note else ''}") from e


class JsonlSink:
    """Append-only jsonl writer that flushes every record and can resume from what is already there.

    Usage:
        sink = JsonlSink(path, key="uid")
        todo = [c for c in clips if c["uid"] not in sink.seen]
        ...
        sink.write(row)
    """

    def __init__(self, path: str | Path, key: str | None = None, fsync_every: int = 25):
        self.path = Path(path)
        self.key = key
        self.fsync_every = max(1, fsync_every)
        self.n = 0
        preflight(self.path)
        self.seen: set = set()
        if key and self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    try: self.seen.add(json.loads(line)[key])
                    except Exception: pass
        self.fh = self.path.open("a")

    def write(self, row: dict[str, Any]) -> None:
        self.fh.write(json.dumps(row) + "\n")
        self.fh.flush()
        self.n += 1
        if self.n % self.fsync_every == 0:
            os.fsync(self.fh.fileno())
        if self.key and self.key in row:
            self.seen.add(row[self.key])

    def extend(self, rows: Iterable[dict[str, Any]]) -> None:
        for r in rows: self.write(r)

    def close(self) -> None:
        try:
            self.fh.flush(); os.fsync(self.fh.fileno())
        finally:
            self.fh.close()

    def __enter__(self): return self

    def __exit__(self, *exc):
        self.close()
        return False
