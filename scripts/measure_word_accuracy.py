#!/usr/bin/env python3
"""Measure word-level accuracy over repeated draws, not one anecdote.

Chatterbox's generate() samples stochastically -- the same prompt and
reference can produce different speech-token draws on different calls. A
single mispronunciation heard once doesn't distinguish "this setting is
broken" from "that was one unlucky sample at whatever the baseline error
rate is." This generates N draws per prompt and transcribes each with Whisper
to get an actual rate.

    python scripts/measure_word_accuracy.py --ref inference/refs/113_000356120.flac \
        --prompts inference/prompts/dummy_prompts.txt --n 5
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def word_diff(reference: str, hypothesis: str) -> tuple[int, int]:
    """(substitutions, reference_word_count) via a sequence alignment.

    Insertions/deletions are real transcription differences too, but
    substitutions are what a mispronunciation looks like -- Whisper hearing a
    different, real word in the same slot.
    """
    ref, hyp = normalize(reference), normalize(hypothesis)
    sm = difflib.SequenceMatcher(a=ref, b=hyp)
    subs = sum(max(a2 - a1, b2 - b1) for tag, a1, a2, b1, b2 in sm.get_opcodes()
              if tag == "replace")
    return subs, len(ref)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5, help="draws per prompt")
    ap.add_argument("--out", type=Path, default=Path("inference/out_measure"))
    ap.add_argument("--adapter-repo", default="aoxo/text2asmr-chatterbox")
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--whisper-model", default="base")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from infer_speech import load_model

    import torch
    import torchaudio as ta
    import whisper

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _ = load_model(device, args.adapter_repo, base_only=False)
    model.prepare_conditionals(str(args.ref), exaggeration=args.exaggeration)
    asr = whisper.load_model(args.whisper_model)

    prompts = [
        l.strip() for l in args.prompts.read_text().splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    args.out.mkdir(parents=True, exist_ok=True)

    total_subs = total_words = 0
    print(f"{len(prompts)} prompts x {args.n} draws, "
          f"cfg={args.cfg_weight} exaggeration={args.exaggeration}\n", flush=True)

    for pi, text in enumerate(prompts):
        for di in range(args.n):
            wav = model.generate(text, exaggeration=args.exaggeration,
                                 cfg_weight=args.cfg_weight)
            dest = args.out / f"p{pi:02d}_d{di}.wav"
            ta.save(str(dest), wav.cpu(), model.sr)
            heard = asr.transcribe(str(dest), language="en", fp16=False)["text"]
            subs, n_words = word_diff(text, heard)
            total_subs += subs
            total_words += n_words
            flag = f"  <-- {subs} word substitution(s)" if subs else ""
            print(f"  p{pi} d{di}: {heard.strip()!r}{flag}", flush=True)

    rate = total_subs / max(total_words, 1) * 100
    print(f"\n{total_subs} substitutions / {total_words} reference words "
          f"= {rate:.1f}% word error rate (substitutions only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
