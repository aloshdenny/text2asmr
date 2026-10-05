#!/usr/bin/env python3
"""Re-cut the human-labelled external trigger clips (class_gate.py --stage cutext picked them) at 48 kHz, up to 10 s:
class_gate cuts 4 s at 16 kHz for audio LLMs, which leaves CLAP (trained on 48 kHz, 10 s windows) half its band.

  python cut_external48.py --clips D:/t2a/gate_ext/clips.json --out D:/t2a/gate_ext48
"""
from __future__ import annotations
import argparse, json, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    url = {json.loads(l)["uid"]: json.loads(l)["url"] for l in open(hf_hub_download("aoxo/t2a-eval-external", "external_eval_manifest.jsonl", repo_type="dataset"))}
    clips = json.loads(a.clips.read_text()); (a.out / "wav").mkdir(parents=True, exist_ok=True)

    def cut(c):
        w = a.out / "wav" / Path(c["wav"]).name
        if not w.exists():
            p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", "10", "-i", url[c["uid"]], "-ac", "1", "-ar", "48000", str(w)], capture_output=True)
            if p.returncode or not w.exists() or w.stat().st_size < 2000: return None
        return {**c, "wav": str(w)}
    with ThreadPoolExecutor(8) as ex: out = [c for c in ex.map(cut, clips) if c]
    (a.out / "clips.json").write_text(json.dumps(out))
    print(f"{len(out)} of {len(clips)} clips at 48 kHz -> {a.out / 'clips.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
