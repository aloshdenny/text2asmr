#!/usr/bin/env python3
"""Publish the sound guide: example clips per label (public bucket "guide") plus guide.json, which the site reads.

Input is a guide plan like build_guided_kit.py's: {"guide": [{"label", "hint", "uids": [...]}]}; the clips are found
by sanitized uid in --clips dirs. Guide clips are meant to be heard by everyone, so they are public.

  export SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=...
  python push_guide.py --plan kit5_plan.json --clips D:/t2a/pool D:/t2a/pool_yt D:/t2a/pool_spray
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_clips import Site, load_env_file, find_audio, sanitize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--clips", type=Path, nargs="+", required=True)
    ap.add_argument("--env-file", type=Path, default=None)
    a = ap.parse_args()
    load_env_file(a.env_file)
    audio = find_audio(a.clips)
    site = Site()
    out = []
    for g in json.loads(a.plan.read_text())["guide"]:
        slug = re.sub(r"[^a-z0-9]+", "-", g["label"].lower()).strip("-")
        names = []
        for i, u in enumerate(g["uids"]):
            f = audio.get(sanitize(u))
            if f is None: print(f"missing audio for {g['label']} example {u}"); continue
            name = f"{slug}-{i + 1}.mp3"
            site.upload("guide", name, f.read_bytes())
            names.append(name)
        out.append({"label": g["label"], "hint": g.get("hint", ""), "clips": names})
    site.upload("guide", "guide.json", json.dumps(out, indent=1).encode(), "application/json")
    print(f"guide: {sum(len(g['clips']) for g in out)} clips for {len(out)} labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
