"""Delete HF cache blobs no snapshot symlink points at.

build_datasets.py unlinks each source file after processing it, but that only
removes the snapshot symlink -- the blob it pointed to stays. An orphaned blob
is therefore precisely a file the build has already consumed, so removing it
is safe while the build is still running.
"""
import os, sys
from pathlib import Path

root = Path.home() / ".cache/huggingface/hub/datasets--aoxo--audios"
blobs = root / "blobs"
if not blobs.exists():
    sys.exit("no blobs dir")

referenced = set()
for snap in (root / "snapshots").rglob("*"):
    if snap.is_symlink():
        try:
            referenced.add(os.path.realpath(snap))
        except OSError:
            pass

freed = kept = 0
for blob in blobs.iterdir():
    if not blob.is_file():
        continue
    size = blob.stat().st_size
    if str(blob.resolve()) in referenced:
        kept += size
        continue
    blob.unlink()
    freed += size

print(f"freed {freed/2**30:.1f} GiB, kept {kept/2**30:.1f} GiB still referenced")
