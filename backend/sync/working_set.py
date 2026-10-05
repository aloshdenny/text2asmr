#!/usr/bin/env python3
"""Keep ASMR Board's storage to the audio people still need to hear (the free plan has 1 GB).

  takedown  clips nobody needs to label now (confident fused labels and no judge yet, or quorum reached): marked
            inactive first, so nobody is served a clip whose audio is going, then their audio is deleted. Rows and
            labels stay.
  reload    taken-down clips people still need (started ones first, then by need), uploaded again from the
            research server's copy to the same object name, until the storage budget is used.
  fill      new low-confidence / weak-label clips from a manifest ({uid, path, kind, uncertainty, info}, as
            push_clips.py --manifest reads), in file order, until the budget is used.

  python working_set.py --env-file D:/t2a/asmrboard.env --clips D:/t2a/pool D:/t2a/pool_yt ... \\
      --takedown --reload --fill D:/t2a/pool_ytdense_all/site_manifest.jsonl --fill-batch 2026-10-04-ytdense --not-core \\
      --budget-mb 900 [--dry-run]
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time, uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_clips import Site, load_env_file, read_audio

MB = 1024 * 1024


def safe(uid: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def size_of(path: str) -> int:
    """File size; clip names are full source paths, past Windows' 260-char limit without the \\\\?\\ prefix."""
    p = os.path.abspath(path)
    if os.name == "nt" and not p.startswith("\\\\?\\"): p = "\\\\?\\" + p
    return os.path.getsize(p)


def upload(site: Site, name: str, path: str) -> None:
    """Upload with a few retries: a single slow response from storage should not end the run."""
    for i in range(4):
        try: site.upload("clips", name, read_audio(Path(path))); return
        except Exception:
            if i == 3: raise
            time.sleep(10 * (i + 1))


def local_files(roots: list[Path]) -> dict[str, str]:
    """Clip audio on this server by safe uid: pool clips (pool_*/clips) and loose dirs of mp3s."""
    out = {}
    for d in roots:
        for f in glob.glob(str(d / "pool_*" / "clips" / "*.mp3")) + glob.glob(str(d / "*.mp3")):
            out.setdefault(Path(f).stem, f)
    return out


def delete_objects(site: Site, names: list[str]) -> None:
    for i in range(0, len(names), 100):
        r = requests.delete(f"{site.url}/storage/v1/object/clips", json={"prefixes": names[i:i + 100]}, headers=site.h, timeout=120)
        r.raise_for_status()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--clips", type=Path, nargs="*", default=[], help="dirs holding the clips' audio (for reload)")
    ap.add_argument("--takedown", action="store_true")
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--fill", type=Path, default=None, help="manifest of new clips to add while there is room")
    ap.add_argument("--fill-batch", default=None)
    ap.add_argument("--not-core", action="store_true", help="fill clips form a verification pool (outside the people ring)")
    ap.add_argument("--target-votes", type=int, default=12)
    ap.add_argument("--budget-mb", type=float, default=900.0, help="stop reloading / filling at this much stored audio")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    load_env_file(a.env_file)
    site = Site()
    used = site.rpc("admin_storage_bytes")
    print(f"storage: {used / MB:.0f} MB used, budget {a.budget_mb:.0f} MB")

    if a.takedown:
        why = {}
        while True:                                          # the API returns at most 1,000 rows a call: page until none left
            rows = site.rpc("admin_takedown_candidates", p_limit=1000)
            for r in rows: why[r["reason"]] = why.get(r["reason"], 0) + 1
            if not rows or a.dry_run: break
            site.rpc("admin_set_active", p_clips=[r["clip_id"] for r in rows], p_active=False)   # stop serving first
            delete_objects(site, [r["audio_path"] for r in rows])
        print(f"takedown: {sum(why.values())} clips {why}" + (" (first page only: dry run)" if a.dry_run else ""))
        used = site.rpc("admin_storage_bytes")
        print(f"  storage now {used / MB:.0f} MB")

    room = a.budget_mb * MB - used
    files = local_files(a.clips) if (a.reload or a.fill) else {}

    if a.reload and room > 0:
        back, missing, seen, full = 0, 0, set(), False
        while not full:                                      # page: each reloaded page drops out of the next call
            cands = [r for r in site.rpc("admin_reload_candidates", p_limit=1000) if r["clip_id"] not in seen]
            if not cands: break
            page = []
            for r in cands:
                seen.add(r["clip_id"])
                f = files.get(safe(r["source_uid"]))
                if not f: missing += 1; continue
                size = size_of(f)
                if size > room: full = True; break
                if not a.dry_run: upload(site, r["audio_path"], f)
                page.append(r["clip_id"]); room -= size
            if page and not a.dry_run: site.rpc("admin_set_active", p_clips=page, p_active=True)
            back += len(page)
            if a.dry_run: break
        print(f"reload: {back} clips back up ({missing} without a local copy)")

    if a.fill and room > 0:
        rows = [json.loads(l) for l in open(a.fill, encoding="utf-8")]
        known = set()
        for i in range(0, len(rows), 500):
            known |= set(site.rpc("admin_known_sources", p_source_uids=[r["uid"] for r in rows[i:i + 500]]))
        new, added = [r for r in rows if r["uid"] not in known], 0
        from concurrent.futures import ThreadPoolExecutor

        def up(r: dict) -> dict:
            name = f"{uuid.uuid4()}.mp3"
            if not a.dry_run: upload(site, name, r["path"])
            return {"audio_path": name, "source_uid": r["uid"], "kind": r.get("kind", "unlabelled"),
                    "uncertainty": round(float(r.get("uncertainty", 0.5)), 4), "target_votes": a.target_votes,
                    "batch": a.fill_batch or "working-set", "info": r.get("info", {}), **({"core": False} if a.not_core else {})}

        with ThreadPoolExecutor(8) as ex:                      # uploads in parallel, registered 200 at a time
            i = 0
            while i < len(new) and room > 0:
                chunk = []
                while i < len(new) and len(chunk) < 200:
                    size = size_of(new[i]["path"])
                    if size > room: room = 0; break
                    chunk.append(new[i]); room -= size; i += 1
                if not chunk: break
                items = list(ex.map(up, chunk))
                if not a.dry_run: site.rpc("admin_add_clips", p_clips=items)
                added += len(items)
        print(f"fill: {added} of {len(new)} new clips added ({len(known)} already queued); room left {room / MB:.0f} MB")

    if not a.dry_run: site.rpc("admin_refresh_progress")
    print(f"WORKING_SET_DONE storage {site.rpc('admin_storage_bytes') / MB:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
