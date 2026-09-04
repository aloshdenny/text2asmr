#!/usr/bin/env python3
"""Pull the gated metadata and report what the corpus actually contains.

Run this first, after ``hf auth login``.  It downloads only the two metadata
CSVs (about 3.6 MB) -- not the 5.8 GB of audio -- so it is cheap to iterate on,
and it answers the questions that decide the training split:

  * what the real trigger vocabulary is, versus the four tags the paper names
  * how many clips are pure speech, pure trigger, or mixed
  * how much of the corpus each backend is therefore responsible for

Usage:
    python scripts/survey_corpus.py [--out reports/corpus.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text2asmr.compose.grammar import parse, survey_vocabulary  # noqa: E402

REPO = "aoxo/text2asmr-uncensored"


def load_rows(filename: str) -> list[dict]:
    """Fetch one metadata CSV from the gated repo.

    The file is LJSpeech-flavoured, so the delimiter is not guaranteed to be a
    comma despite the extension; sniff it rather than assume.
    """
    import csv

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, filename, repo_type="dataset")
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    sample = "\n".join(text.splitlines()[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "|" if "|" in sample else ","

    reader = csv.reader(text.splitlines(), delimiter=delimiter)
    rows = [r for r in reader if r]
    print(f"  {filename}: {len(rows)} rows, delimiter {delimiter!r}, "
          f"{len(rows[0])} columns")
    return rows


def transcript_of(row: list[str]) -> str:
    """Pick the transcript column: the widest text field in the row.

    LJSpeech metadata is ``id|transcript|normalised``, but this corpus was
    hand-assembled, so locate the transcript by shape instead of by index.
    """
    return max(row[1:], key=len) if len(row) > 1 else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/corpus.json"))
    ap.add_argument("--file", default="metadata.csv")
    args = ap.parse_args()

    try:
        rows = load_rows(args.file)
    except Exception as exc:  # noqa: BLE001 - surface auth errors legibly
        print(f"could not read {args.file}: {exc}", file=sys.stderr)
        print("\nIf this is a 401/403, the dataset is gated -- run "
              "`hf auth login` first.", file=sys.stderr)
        return 1

    transcripts = [transcript_of(r) for r in rows]
    triggers, intensities = survey_vocabulary(transcripts)

    routing: Counter = Counter()
    for transcript in transcripts:
        script = parse(transcript)
        if script.is_pure_speech:
            routing["speech_only"] += 1
        elif script.is_pure_trigger:
            routing["trigger_only"] += 1
        elif script.segments:
            routing["mixed"] += 1
        else:
            routing["empty"] += 1

    report = {
        "repo": REPO,
        "file": args.file,
        "clips": len(rows),
        "routing": dict(routing),
        "intensities": dict(intensities),
        "trigger_vocab_size": len(triggers),
        "triggers": dict(triggers.most_common()),
    }

    print(f"\nclips              {len(rows)}")
    print("routing            " + ", ".join(f"{k}={v}" for k, v in routing.most_common()))
    print(f"intensity tags     {dict(intensities)}")
    print(f"distinct triggers  {len(triggers)}")
    print("\ntop triggers:")
    for name, count in triggers.most_common(30):
        print(f"  {count:6d}  {name}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
