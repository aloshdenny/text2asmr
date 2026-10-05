#!/usr/bin/env python3
"""Queue Real-or-AI items (scripts/rai_items.py finish) on ASMR Board: upload each mp3 under an opaque name, exactly
like a clip, and register it with its answer, which only the server keeps.

  python push_rai.py --manifest D:/t2a/rai/manifest.jsonl --env-file D:/t2a/asmrboard.env
"""
from __future__ import annotations
import argparse, json, sys, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_clips import Site, load_env_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--env-file", type=Path, default=None)
    a = ap.parse_args()
    load_env_file(a.env_file)
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    site = Site()

    def up(r: dict) -> dict:
        name = f"{uuid.uuid4()}.mp3"
        site.upload("clips", name, Path(r["file"]).read_bytes())
        return {"audio_path": name, "is_ai": r["is_ai"], "voice": r["voice"], "model": r.get("model"), "source": r["source"]}

    added = 0
    with ThreadPoolExecutor(8) as ex:
        for i in range(0, len(rows), 100):                   # register a chunk at a time, right after its upload
            added += site.rpc("admin_add_rai_items", p_items=list(ex.map(up, rows[i:i + 100])))
    print(f"added {added} Real-or-AI items ({sum(r['is_ai'] for r in rows)} AI, {sum(not r['is_ai'] for r in rows)} real)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
