#!/usr/bin/env python3
"""Upload the built audio to the Hub as tar shards instead of loose files.

Uploading ~165,000 individual FLACs got the repo rate-limited: HF's LFS/xet
endpoint started returning 429 and the sync spun for hours transferring
nothing. Per-file overhead, not bandwidth, was the binding constraint -- the
Hub warns about folders over 10,000 entries for exactly this reason.

Shards fix it by turning 165,000 requests into ~75. The layout is WebDataset
style (a flat tar of `<uid>.flac`), which `datasets` can stream directly, and
each shard is built, uploaded, and deleted in turn so peak disk stays at one
shard rather than a second copy of the corpus.

Resumable: shards already present in the repo are skipped, so an interrupted
run costs at most the shard in flight.

    python scripts/shard_upload.py --out ~/text2asmr/out --repo aoxo/text2asmr-segments
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path


def read_names(meta: Path) -> list[str]:
    """File names from metadata.jsonl -- the source of truth, not the directory.

    Audio is written before its metadata row, so a directory listing can
    contain orphans from an interrupted run. Rows also repeat when a restart
    re-processes the file that was in flight.
    """
    seen: set[str] = set()
    names: list[str] = []
    for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            name = json.loads(line)["file_name"]
        except (json.JSONDecodeError, KeyError):
            continue
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / "text2asmr/out")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--shard-mb", type=int, default=200)
    ap.add_argument("--kinds", default="speech,triggers")
    ap.add_argument("--work", type=Path, default=Path("/tmp/text2asmr-shards"))
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        existing = set(api.list_repo_files(args.repo, repo_type="dataset"))
    except Exception as exc:  # noqa: BLE001
        print(f"cannot read {args.repo}: {exc}", file=sys.stderr)
        return 1
    args.work.mkdir(parents=True, exist_ok=True)

    for kind in args.kinds.split(","):
        kind = kind.strip()
        root = args.out / kind
        meta = root / "metadata.jsonl"
        if not meta.exists():
            print(f"{kind}: no metadata.jsonl, skipping")
            continue

        names = read_names(meta)
        print(f"\n{kind}: {len(names):,} files from metadata")

        shard_idx, cur, cur_bytes, shards = 0, [], 0, []
        limit = args.shard_mb * 1024 * 1024
        for name in names:
            p = root / name
            try:
                size = p.stat().st_size
            except OSError:
                continue  # row present, audio gone
            cur.append((name, p))
            cur_bytes += size
            if cur_bytes >= limit:
                shards.append(cur)
                cur, cur_bytes = [], 0
                shard_idx += 1
        if cur:
            shards.append(cur)

        print(f"{kind}: {len(shards)} shards of ~{args.shard_mb}MB")
        t0 = time.time()
        uploaded = skipped = 0

        for i, members in enumerate(shards):
            remote = f"{kind}/shards/{kind}-{i:05d}.tar"
            if remote in existing:
                skipped += 1
                continue

            local = args.work / f"{kind}-{i:05d}.tar"
            with tarfile.open(local, "w") as tf:
                for name, path in members:
                    tf.add(path, arcname=name)

            try:
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=remote,
                    repo_id=args.repo,
                    repo_type="dataset",
                )
                uploaded += 1
            except Exception as exc:  # noqa: BLE001
                # A 429 here means back off rather than hammer the endpoint
                # that just rate-limited us.
                print(f"  {remote}: {type(exc).__name__}: {str(exc)[:120]}")
                local.unlink(missing_ok=True)
                print("  backing off 60s")
                time.sleep(60)
                continue
            finally:
                local.unlink(missing_ok=True)

            done = uploaded + skipped
            rate = uploaded / max(time.time() - t0, 1e-9) * 3600
            eta = (len(shards) - done) / max(rate, 1e-9)
            print(f"  [{done}/{len(shards)}] {remote}  "
                  f"{rate:.0f} shards/h  eta {eta:.1f}h", flush=True)

        print(f"{kind}: uploaded {uploaded}, skipped {skipped} already present")

    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
