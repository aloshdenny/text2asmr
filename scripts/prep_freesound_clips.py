#!/usr/bin/env python3
"""Make Freesound previews look like every other pool clip, and keep non-CC0 audio off the public site.

Previews are up to 30 s of 128 kbps stereo; the judges, CLAP and the crowd site all work on 6 s mono 24 kHz 48 kbps
clips. Each preview is replaced by its loudest 6 s window (the sound is usually somewhere in the middle, with silence
around it), re-encoded in the pool format. Rows get the window's start.

Licences: CC-BY needs attribution wherever the audio is redistributed and CC-BY-NC forbids commercial use. The crowd
site shows no attribution, so only CC0 clips may be served there: every other uid goes into --site-exclude, which
rank_fused_for_people.py --exclude reads. All clips stay usable for labelling, and each row keeps its licence and
author so training can filter too.

  python prep_freesound_clips.py --pool D:/t2a/pool_freesound --site-exclude D:/t2a/pool_freesound/not_cc0.txt
"""
from __future__ import annotations
import argparse, json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SR, WIN = 24000, 6.0


def safe(uid: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def loudest_start(path: Path) -> float:
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    y = np.frombuffer(pcm, np.float32)
    n = int(WIN * SR)
    if len(y) <= n: return 0.0
    e = np.convolve(y * y, np.ones(n, np.float32), "valid")[:: SR // 10]   # 0.1 s steps
    return round(float(np.argmax(e)) * 0.1, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--site-exclude", type=Path, required=True)
    a = ap.parse_args()
    clips = a.pool / "pool_freesound" / "clips"
    rows = [json.loads(l) for l in open(a.pool / "pool.jsonl", encoding="utf-8")]

    def one(r):
        f = clips / f"{safe(r['uid'])}.mp3"
        if not f.exists() or r.get("prepped"): return r
        try:
            t = loudest_start(f)
            tmp = f.with_suffix(".tmp.mp3")
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{t:.2f}", "-t", str(WIN), "-i", str(f),
                            "-ac", "1", "-ar", str(SR), "-b:a", "48k", str(tmp)], check=True, capture_output=True)
            os.replace(tmp, f)
            r.update(start=t, prepped=True)
        except Exception as e:
            r["prep_error"] = type(e).__name__
        return r

    with ThreadPoolExecutor(os.cpu_count() or 8) as ex: rows = list(ex.map(one, rows))
    (a.pool / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    not_cc0 = [r["uid"] for r in rows if "publicdomain/zero" not in (r.get("license") or "")]
    a.site_exclude.write_text("".join(u + "\n" for u in not_cc0), encoding="utf-8")
    print(f"PREP_DONE {sum(1 for r in rows if r.get('prepped'))} clips cut to {WIN:.0f} s; "
          f"{len(rows) - len(not_cc0)} CC0 (site-eligible), {len(not_cc0)} kept off the site", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
