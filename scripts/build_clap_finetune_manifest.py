#!/usr/bin/env python3
"""Build a CLAP fine-tune manifest from Gemini-labeled rows.

Sources:
  - label_tool/clap_train_audios2*.jsonl  (audios2 cuts: source+start+duration)
  - label_tool/gemini_retag.jsonl         (segments FLACs on aoxo/text2asmr-segments)

Writes label_tool/clap_finetune_manifest.jsonl with one row per training example.
Only in-ontology triggers (plus newly added keys) are kept; reject/OOD dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from text2asmr.data.ontology import BY_KEY, intensity_from_loudness  # noqa: E402

SEGMENTS_REPO = "aoxo/text2asmr-segments"
AUDIOS2_REPO = "aoxo/audios2"


def clap_texts(trigger: str, intensity: str = "soft") -> list[str]:
    primary = f"ASMR {intensity} {trigger}, close-mic binaural, no speech"
    probes = list(BY_KEY[trigger].probes) if trigger in BY_KEY else []
    out, seen = [], set()
    for t in [primary, *probes, f"The sound of {trigger}"]:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def from_clap_train(root: Path) -> list[dict]:
    paths = [root / "clap_train_audios2.jsonl"]
    paths += sorted(root.glob("clap_train_audios2*.jsonl"))
    out, seen = [], set()
    for p in paths:
        for r in load_jsonl(p):
            uid = r.get("uid") or f"{r.get('source')}_{r.get('start')}"
            trig = r.get("trigger")
            if not trig or trig not in BY_KEY or uid in seen:
                continue
            seen.add(uid)
            out.append(
                {
                    "uid": uid,
                    "trigger": trig,
                    "text": r.get("text") or clap_texts(trig, r.get("intensity") or "soft"),
                    "caption": r.get("caption") or clap_texts(trig)[0],
                    "audio": {
                        "kind": "audios2_cut",
                        "repo": AUDIOS2_REPO,
                        "source": r["source"],
                        "start": float(r["start"]),
                        "duration": float(r["duration"]),
                    },
                    "labeler": r.get("labeler") or "gemini",
                }
            )
    return out


def from_gemini_retag(root: Path) -> list[dict]:
    out, seen = [], set()
    for r in load_jsonl(root / "gemini_retag.jsonl"):
        trig = r.get("gemini_label")
        fn = r.get("file_name")
        if not fn or not trig or trig == "reject" or trig not in BY_KEY:
            continue
        if fn in seen:
            continue
        seen.add(fn)
        intensity, _ = intensity_from_loudness(-35.0)  # unknown RMS; soft default via texts
        texts = clap_texts(trig, "soft")
        out.append(
            {
                "uid": f"segments/{fn}",
                "trigger": trig,
                "text": texts,
                "caption": texts[0],
                "audio": {
                    "kind": "segments_flac",
                    "repo": SEGMENTS_REPO,
                    "path": f"triggers/{fn}",
                },
                "labeler": "gemini-retag",
                "clap_label": r.get("clap_label"),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "label_tool" / "clap_finetune_manifest.jsonl")
    ap.add_argument("--skip-retag", action="store_true")
    ap.add_argument("--skip-audios2", action="store_true")
    args = ap.parse_args()

    root = HERE / "label_tool"
    rows: list[dict] = []
    if not args.skip_audios2:
        rows.extend(from_clap_train(root))
    if not args.skip_retag:
        rows.extend(from_gemini_retag(root))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter

    c = Counter(r["trigger"] for r in rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    print("triggers:", c.most_common(12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
