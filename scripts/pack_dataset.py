#!/usr/bin/env python3
"""Package the built segments into HF datasets and optionally push them.

Reads the two ``metadata.jsonl`` files the builder appends to and emits sharded
parquet with the audio inlined, which is what `datasets` loads fastest and what
the training scripts expect. Two datasets are produced because the backends
want different things:

  speech   24 kHz mono + transcript      -> TTS finetune
  triggers 48 kHz stereo + caption/tag   -> audio-diffusion finetune

Safety properties that matter here:

  * ``metadata.jsonl`` is the source of truth, not the FLAC directory. The
    builder writes the audio file first and the metadata row second, so a kill
    between the two leaves an orphan FLAC. Orphans are ignored.
  * Rows are de-duplicated by ``file_name``. A restart re-processes the source
    file that was in flight, which can append rows already present.
  * A held-out split is carved by *source file*, never by segment. Splitting by
    segment would put clips from the same recording -- often the same speaker,
    seconds apart -- on both sides, and the eval number would be meaningless.

    python scripts/pack_dataset.py --out ~/t2a/out --dest ~/t2a/packed
    python scripts/pack_dataset.py --out ~/t2a/out --dest ~/t2a/packed \
        --push aoxo/text2asmr-speech --private
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def read_rows(meta: Path) -> list[dict]:
    """Load metadata.jsonl, dropping duplicates and rows whose audio is gone."""
    if not meta.exists():
        return []
    seen: set[str] = set()
    rows: list[dict] = []
    dupes = missing = bad = 0
    for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A kill mid-write can leave a torn final line.
            bad += 1
            continue
        name = row.get("file_name")
        if not name or name in seen:
            dupes += 1
            continue
        if not (meta.parent / name).exists():
            missing += 1
            continue
        seen.add(name)
        rows.append(row)
    print(f"  {meta.parent.name}: {len(rows)} rows "
          f"({dupes} dupes, {missing} missing audio, {bad} torn lines)")
    return rows


def split_by_source(rows: list[dict], eval_frac: float) -> tuple[list, list]:
    """Hold out whole source recordings, not individual segments.

    Segments from one recording share a speaker, a room and a mic, so a
    segment-level split leaks the eval condition into training.
    """
    sources = sorted({r.get("source", "") for r in rows})
    n_eval = max(1, int(len(sources) * eval_frac)) if sources else 0
    # Deterministic: take a strided sample so the held-out set is spread across
    # the corpus rather than clustered at one end.
    stride = max(len(sources) // n_eval, 1) if n_eval else 1
    held = set(sources[::stride][:n_eval])
    train = [r for r in rows if r.get("source") not in held]
    evl = [r for r in rows if r.get("source") in held]
    return train, evl


def build(rows: list[dict], root: Path, kind: str):
    """Turn metadata rows into a `datasets.Dataset` with decoded audio column."""
    from datasets import Audio, Dataset

    for r in rows:
        r["audio"] = str(root / r["file_name"])
    ds = Dataset.from_list(rows)
    # Let `datasets` own decoding so the parquet carries the bytes and the
    # training script never touches the filesystem layout.
    return ds.cast_column("audio", Audio(sampling_rate=None))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="builder output dir")
    ap.add_argument("--dest", type=Path, required=True, help="where to save")
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--push-speech", default="", help="hub repo id for speech")
    ap.add_argument("--push-triggers", default="", help="hub repo id for triggers")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap rows, for smoke tests")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    for kind, push_to in (("speech", args.push_speech),
                          ("triggers", args.push_triggers)):
        root = args.out / kind
        rows = read_rows(root / "metadata.jsonl")
        if not rows:
            print(f"  {kind}: nothing to pack, skipping")
            continue
        if args.limit:
            rows = rows[: args.limit]

        train, evl = split_by_source(rows, args.eval_frac)
        print(f"  {kind}: {len(train)} train / {len(evl)} eval "
              f"(held-out sources: {len({r.get('source') for r in evl})})")

        from datasets import DatasetDict

        dd = DatasetDict({"train": build(train, root, kind)})
        if evl:
            dd["eval"] = build(evl, root, kind)

        target = args.dest / kind
        dd.save_to_disk(str(target))
        print(f"  {kind}: saved -> {target}")

        stats = {"train": len(train), "eval": len(evl),
                 "hours": round(sum(r.get("duration", 0) for r in rows) / 3600, 2)}
        if kind == "triggers":
            stats["per_trigger"] = dict(
                Counter(r.get("trigger", "?") for r in rows).most_common()
            )
            stats["per_intensity"] = dict(
                Counter(r.get("intensity", "?") for r in rows).most_common()
            )
        report[kind] = stats

        if push_to:
            print(f"  {kind}: pushing to {push_to} (private={args.private})")
            dd.push_to_hub(push_to, private=args.private)

    (args.dest / "report.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
