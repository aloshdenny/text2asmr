#!/usr/bin/env python3
"""Pull the trigger shards from the Hub and unpack them flat for stable-audio-tools.

`audio_dir` datasets in stable-audio-tools recursively scan a directory for
audio files -- there is no shard-aware loader for a local `audio_dir`, so the
tars from :mod:`shard_upload` are downloaded and extracted, then deleted, one
at a time to keep peak disk at one shard rather than a second full copy.

    python scripts/fetch_triggers.py --out /workspace/out/triggers
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", default="aoxo/text2asmr-segments")
    ap.add_argument("--min-clips", type=int, default=15000,
                    help="sanity floor; the corpus has ~20,340 trigger clips")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    args.out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files(args.repo, repo_type="dataset")
    shards = sorted(f for f in files
                    if f.startswith("triggers/shards/") and f.endswith(".tar"))
    if not shards:
        raise SystemExit(f"no trigger shards found in {args.repo}")
    print(f"{len(shards)} shards", flush=True)

    for i, name in enumerate(shards, 1):
        path = Path(hf_hub_download(args.repo, name, repo_type="dataset"))
        # Explicit "r:" (no compression, no auto-detect): shards are always
        # plain tars, and auto-detect mode speculatively probes bz2 first on
        # every open regardless of content. That probe is harmless where a
        # real _bz2 exists, but on a venv carrying the stub from the
        # torchvision/CLAP import fix (this box), the stub's NotImplementedError
        # isn't one of the exceptions tarfile's auto-detect catches, so it
        # aborts instead of falling through to plain-tar reading.
        with tarfile.open(path, "r:") as tf:
            tf.extractall(args.out)
        path.unlink(missing_ok=True)
        print(f"  [{i}/{len(shards)}] {name}", flush=True)

    meta = hf_hub_download(args.repo, "triggers/metadata.jsonl",
                           repo_type="dataset")
    (args.out / "metadata.jsonl").write_bytes(Path(meta).read_bytes())

    n = len(list(args.out.glob("*.flac")))
    print(f"unpacked {n:,} flac files -> {args.out}")
    if n < args.min_clips:
        raise SystemExit(f"only {n} clips unpacked, expected >= {args.min_clips} "
                         f"-- a download likely failed partway")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
