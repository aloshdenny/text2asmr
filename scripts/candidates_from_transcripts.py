#!/usr/bin/env python3
"""Build gap-clip candidates (same schema as label_audios2_qwen3 candidates.jsonl) from Whisper alignment JSONs on the Hub."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import hf_hub_download
from text2asmr.data.segment import load_alignment, split_alignment
ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--files", required=True, help="text file: one m4a path per line"); ap.add_argument("--out", required=True)
a = ap.parse_args(); n = 0
with open(a.out, "a") as f:
    for src in [l.strip() for l in open(a.files) if l.strip()]:
        try: entries = load_alignment(hf_hub_download(a.repo, src + ".json", repo_type="dataset"))
        except Exception as e: print("skip", src[:60], type(e).__name__); continue
        for sp in split_alignment(entries, src):
            if sp.kind != "trigger_candidate": continue
            f.write(json.dumps({"uid": sp.uid, "source": src, "repo": a.repo, "start": round(sp.start, 3), "duration": round(sp.duration, 3), "old": None, "cut_start": round(max(0.0, sp.start - 1.0), 3), "cut_duration": round(min(8.0, sp.duration + 2.0), 3)}) + "\n"); n += 1
print("candidates", n)
