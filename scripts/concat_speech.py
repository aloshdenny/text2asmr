#!/usr/bin/env python3
"""Build a long-form speech set by concatenating adjacent segments.

The model was trained on phrases averaging 2.9s (median) because
``segment.py`` splits on any pause over ``PHRASE_GAP_S``, which is correct for
transcription but means T3 never saw an utterance longer than a few seconds --
so it learned to stop early, and does, even when a prompt is long. Generation
length is bounded by what training length looked like, not by anything in the
architecture.

This concatenates consecutive same-source segments (in time order) into
longer training examples, purely as post-processing over the FLACs already
extracted by build_datasets.py -- no source audio, no GPU, no re-download.
A short silence is inserted at each join rather than a hard cut, since a
zero-gap splice reads as a click and is not what a real pause sounds like.

Segments are joined regardless of what happened between them (a trigger
sound, a long pause) because the goal is a realistic long-utterance length
distribution, not a verbatim transcript -- the join silence stands in for
whatever was cut. If you need the actual trigger audio preserved in the gap,
that is a different pipeline (interleaving speech+trigger spans), not this.

    python scripts/concat_speech.py --out ~/text2asmr/out --target-s 15 --max-s 20
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / "text2asmr/out")
    ap.add_argument("--dest-name", default="speech_long",
                    help="output subdir under --out")
    ap.add_argument("--target-s", type=float, default=15.0,
                    help="stop adding segments to a group once it reaches this")
    ap.add_argument("--max-s", type=float, default=20.0,
                    help="hard cap; matches segment.py's MAX_SPEECH_S so "
                         "downstream token budgets stay consistent")
    ap.add_argument("--join-silence-ms", type=int, default=250,
                    help="silence inserted at each splice, standing in for "
                         "whatever occurred in the original gap")
    ap.add_argument("--min-segments", type=int, default=2,
                    help="drop groups smaller than this -- a lone segment "
                         "isn't a concatenation, it's just the original clip")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf

    src = args.out / "speech"
    meta_path = src / "metadata.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"no metadata at {meta_path}")

    rows = []
    for line in meta_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    for source_rows in by_source.values():
        source_rows.sort(key=lambda r: r["start"])

    dest = args.out / args.dest_name
    dest.mkdir(parents=True, exist_ok=True)
    out_meta = (dest / "metadata.jsonl").open("w")

    sr = None
    n_groups = n_dropped_short = n_dropped_missing = 0
    total_s = 0.0

    for source, source_rows in by_source.items():
        # A mutable-default-argument version of this (`def flush(group=group)`)
        # is a trap: the default is bound once, at definition time, so every
        # later `group = []` rebind leaves flush() writing the same frozen,
        # stale list forever. `nonlocal` plus in-place `.clear()` avoids that
        # -- there is only ever one list object for this source.
        group: list[dict] = []
        group_dur = 0.0

        def flush():
            nonlocal n_groups, n_dropped_short, n_dropped_missing, total_s, sr
            if len(group) < args.min_segments:
                if group:
                    n_dropped_short += 1
                group.clear()
                return
            chunks = []
            for r in group:
                p = src / r["file_name"]
                if not p.exists():
                    n_dropped_missing += 1
                    group.clear()
                    return
                audio, this_sr = sf.read(p, dtype="float32")
                sr = sr or this_sr
                if this_sr != sr:
                    n_dropped_missing += 1  # mixed rates, skip rather than resample silently
                    group.clear()
                    return
                chunks.append(audio)

            silence = np.zeros(int(sr * args.join_silence_ms / 1000),
                               dtype=np.float32)
            joined = chunks[0]
            for c in chunks[1:]:
                joined = np.concatenate([joined, silence, c])
            text = " ".join(r["text"] for r in group)

            uid = f"{source}_{int(group[0]['start']*1000):09d}_x{len(group)}"
            name = f"{uid}.flac"
            sf.write(dest / name, joined, sr)
            dur = len(joined) / sr
            out_meta.write(json.dumps({
                "file_name": name, "text": text, "source": source,
                "start": group[0]["start"], "duration": round(dur, 3),
                "n_segments": len(group),
            }) + "\n")
            n_groups += 1
            total_s += dur
            group.clear()

        for r in source_rows:
            projected = group_dur + r["duration"] + args.join_silence_ms / 1000
            if group and projected > args.max_s:
                flush()
                group_dur = 0.0
            group.append(r)
            group_dur += r["duration"] + (args.join_silence_ms / 1000 if len(group) > 1 else 0)
            if group_dur >= args.target_s:
                flush()
                group_dur = 0.0
        flush()

    out_meta.close()
    print(f"{n_groups} concatenated examples, {total_s/3600:.2f} h total, "
          f"mean {total_s/max(n_groups,1):.1f}s per example")
    print(f"dropped: {n_dropped_short} too-short groups, "
          f"{n_dropped_missing} with missing/mismatched audio")
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
