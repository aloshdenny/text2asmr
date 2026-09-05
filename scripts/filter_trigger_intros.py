#!/usr/bin/env python3
"""Drop trigger clips that fall in a source video's intro region.

segment.py's trigger-candidate detection treats any ASR word-gap as a
possible trigger, which can't tell a real silent gap ("brushing, no talking")
apart from a generic intro music bed at the start of a video ("no talking,"
also true of the sting before the ASMRtist starts). Listening to a sample
confirmed this: it sounded like generic intro music. Each clip's metadata
already records ``start`` (seconds into its source file, see
build_datasets.py), so filtering is a cheap local operation on the already-
extracted corpus -- no re-decoding of source audio needed.

    python scripts/filter_trigger_intros.py --dir ~/t2a/out/triggers --intro-skip-s 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--intro-skip-s", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be dropped without deleting anything")
    args = ap.parse_args()

    meta_path = args.dir / "metadata.jsonl"
    rows = []
    for line in meta_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    kept, dropped = [], []
    for row in rows:
        if row.get("start", 0.0) < args.intro_skip_s:
            dropped.append(row)
        else:
            kept.append(row)

    print(f"{len(rows)} total, {len(dropped)} in the first {args.intro_skip_s}s "
          f"of their source ({100 * len(dropped) / max(len(rows), 1):.1f}%), "
          f"{len(kept)} kept")

    if args.dry_run:
        return 0

    for row in dropped:
        (args.dir / row["file_name"]).unlink(missing_ok=True)

    meta_path.write_text(
        "".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8"
    )
    print(f"-> {meta_path} rewritten, {len(dropped)} clips deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
