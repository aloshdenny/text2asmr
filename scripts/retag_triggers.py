#!/usr/bin/env python3
"""Re-tag already-extracted trigger clips against the expanded ontology.

The original tagging happened at build time, against whatever TRIGGERS list
existed in ontology.py then. Extending the ontology (finer breathing/rustling
classes, clinking, kissing, footsteps) doesn't require re-decoding source
audio -- CLAP tagging is zero-shot over whatever clip you hand it, so this
re-embeds the clips already sitting in out/triggers/ (or downloaded shards)
against the current, larger probe set, and adds a spatial descriptor
(panned left/right/centered/moving) computed directly from the stereo audio.

Writes a new metadata.jsonl rather than overwriting in place, so the original
tags are never destroyed by a bad re-tagging run.

    python scripts/retag_triggers.py --triggers-dir ~/t2a/out/triggers \
        --out ~/t2a/out/triggers/metadata_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triggers-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: metadata_v2.jsonl next to the input")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import soundfile as sf
    import torch

    from t2a.data.ontology import MARGIN, all_probes, spatial_descriptor

    device = args.device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}", flush=True)

    meta_path = args.triggers_dir / "metadata.jsonl"
    rows = []
    for line in meta_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} rows", flush=True)

    from transformers import ClapModel, ClapProcessor

    CLAP = "laion/clap-htsat-fused"
    model = ClapModel.from_pretrained(CLAP).to(device).eval()
    processor = ClapProcessor.from_pretrained(CLAP)

    texts, owners = all_probes()
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        text_emb = model.get_text_features(
            **{k: v.to(device) for k, v in inputs.items()}
        )
        text_emb = getattr(text_emb, "pooler_output", text_emb)
    pos = [i for i, o in enumerate(owners) if o is not None]
    neg = [i for i, o in enumerate(owners) if o is None]

    out_path = args.out or (args.triggers_dir / "metadata_v2.jsonl")
    out = out_path.open("w")

    changed = 0
    for i in range(0, len(rows), args.batch):
        chunk = rows[i : i + args.batch]
        mono_clips, stereo_clips = [], []
        keep_idx = []
        for j, r in enumerate(chunk):
            p = args.triggers_dir / r["file_name"]
            if not p.exists():
                continue
            stereo, sr = sf.read(p, dtype="float32", always_2d=True)
            stereo = stereo.T  # (channels, samples)
            mono = stereo.mean(axis=0) if stereo.shape[0] > 1 else stereo[0]
            mono_clips.append(mono)
            stereo_clips.append(stereo)
            keep_idx.append(j)

        if not mono_clips:
            continue

        inputs = processor(audio=mono_clips, sampling_rate=sr,
                           return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = model.get_audio_features(
                **{k: v.to(device) for k, v in inputs.items()}
            )
            emb = getattr(emb, "pooler_output", emb)
            scores = emb @ text_emb.T

        for k, row_idx in enumerate(keep_idx):
            r = dict(chunk[row_idx])
            row = scores[k]
            best_pos, pos_i = row[pos].max(dim=0)
            best_neg = row[neg].max()
            if (best_pos - best_neg).item() >= MARGIN:
                new_trigger = owners[pos[int(pos_i)]]
                if new_trigger != r.get("trigger"):
                    changed += 1
                r["trigger"] = new_trigger
                r["clap_confidence_v2"] = round(float(best_pos), 4)

            spatial = spatial_descriptor(stereo_clips[k])
            r["spatial"] = spatial
            lead = f"{r.get('intensity', '')} ".strip()
            spatial_prefix = "" if spatial == "centered" else f"{spatial} "
            r["caption"] = (f"ASMR {spatial_prefix}{lead} {r['trigger']}, "
                            f"close-mic binaural, no speech").replace("  ", " ")
            out.write(json.dumps(r) + "\n")

        if (i // args.batch) % 20 == 0:
            print(f"  {i+len(chunk)}/{len(rows)}", flush=True)

    out.close()
    print(f"\n{changed} clips got a different trigger label under the "
          f"expanded ontology -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
