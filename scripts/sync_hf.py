#!/usr/bin/env python3
"""Back up built data and training checkpoints to the Hub, resumably.

The box this runs on has rebooted twice mid-job and dropped off the network
once, and everything built so far exists only on its local disk. This pushes
that state somewhere durable so a dead box costs time, not work.

``upload_large_folder`` is used rather than ``upload_folder`` because it is
built for exactly this: many files, multi-threaded, and resumable -- it tracks
what already landed and re-uploads only the rest, so an interrupted sync
restarts cheaply instead of from zero.

Repos are created **private** by default. The corpus carries identifiable
voices from YouTube ASMR creators, so a public default would be the wrong way
round; opening one up should be a deliberate act.

    # one-shot backup of the built segments
    python scripts/sync_hf.py data --out ~/text2asmr/out --repo aoxo/text2asmr-segments

    # push a training checkpoint
    python scripts/sync_hf.py ckpt --path ~/text2asmr/ckpt/speech/step-500 \
        --repo aoxo/text2asmr-chatterbox --revision step-500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def ensure_repo(api, repo_id: str, repo_type: str, private: bool) -> None:
    """Create the repo if absent. Idempotent, so callers can just call it."""
    from huggingface_hub.utils import HfHubHTTPError

    try:
        api.repo_info(repo_id, repo_type=repo_type)
        print(f"  repo exists: {repo_id}")
    except HfHubHTTPError:
        api.create_repo(repo_id, repo_type=repo_type, private=private,
                        exist_ok=True)
        print(f"  created {'private ' if private else ''}{repo_id}")


def check_auth() -> str:
    """Fail early and legibly rather than deep inside an upload."""
    from huggingface_hub import whoami

    try:
        return whoami()["name"]
    except Exception:
        print("Not authenticated to the Hub.\n"
              "Run `hf auth login` on this machine first -- the token is stored\n"
              "in the local keyring and never needs to be passed to this script.",
              file=sys.stderr)
        raise SystemExit(2)


def sync_data(args) -> int:
    from huggingface_hub import HfApi

    user = check_auth()
    print(f"authenticated as {user}")
    api = HfApi()
    ensure_repo(api, args.repo, "dataset", not args.public)

    root = args.out
    if not root.exists():
        print(f"nothing at {root}", file=sys.stderr)
        return 1

    # A manifest makes the upload self-describing and lets a later run tell at
    # a glance what a snapshot contained without downloading the audio.
    manifest = {"synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for kind in ("speech", "triggers"):
        meta = root / kind / "metadata.jsonl"
        if meta.exists():
            rows = sum(1 for line in meta.open() if line.strip())
            manifest[kind] = {"rows": rows}
    state = root / "state.json"
    if state.exists():
        st = json.loads(state.read_text())
        manifest["source_files_done"] = len(st.get("done", []))
        manifest["speech_hours"] = round(st.get("speech_s", 0) / 3600, 2)
        manifest["trigger_hours"] = round(st.get("trigger_s", 0) / 3600, 2)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("  manifest:", json.dumps(manifest))

    print(f"  uploading {root} -> {args.repo} (resumable)")
    api.upload_large_folder(
        folder_path=str(root),
        repo_id=args.repo,
        repo_type="dataset",
        # Audio is already FLAC; let the hub store it as-is.
        allow_patterns=["**/*.flac", "**/*.jsonl", "*.json"],
        num_workers=args.workers,
    )
    print("  done")
    return 0


def sync_ckpt(args) -> int:
    from huggingface_hub import HfApi

    user = check_auth()
    print(f"authenticated as {user}")
    api = HfApi()
    ensure_repo(api, args.repo, "model", not args.public)

    if not args.path.exists():
        print(f"nothing at {args.path}", file=sys.stderr)
        return 1

    # Checkpoints go on their own branch so a later one never clobbers an
    # earlier one, and any of them can be resumed from.
    revision = args.revision or args.path.name
    try:
        api.create_branch(args.repo, branch=revision, repo_type="model",
                          exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - non-fatal, main branch still works
        print(f"  branch {revision}: {type(exc).__name__} (continuing on main)")
        revision = None

    print(f"  uploading {args.path} -> {args.repo}"
          f"{f' @ {revision}' if revision else ''}")
    api.upload_folder(
        folder_path=str(args.path),
        repo_id=args.repo,
        repo_type="model",
        revision=revision,
        commit_message=f"checkpoint {args.path.name}",
    )
    print("  done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="back up built segments")
    d.add_argument("--out", type=Path, default=Path.home() / "text2asmr/out")
    d.add_argument("--repo", required=True)
    d.add_argument("--public", action="store_true")
    d.add_argument("--workers", type=int, default=8)
    d.set_defaults(fn=sync_data)

    c = sub.add_parser("ckpt", help="back up a training checkpoint")
    c.add_argument("--path", type=Path, required=True)
    c.add_argument("--repo", required=True)
    c.add_argument("--revision", default="")
    c.add_argument("--public", action="store_true")
    c.set_defaults(fn=sync_ckpt)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
