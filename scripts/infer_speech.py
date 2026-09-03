#!/usr/bin/env python3
"""Generate ASMR speech with the finetuned Chatterbox LoRA adapter.

Loads stock ChatterboxTTS, attaches the T2A LoRA adapter from
aoxo/text2asmr-chatterbox on top of it, and synthesises each prompt in
prompts.txt using a reference clip for voice/delivery conditioning.

Only T3 (text -> speech tokens) was finetuned; S3Gen and the voice encoder are
the stock pretrained weights, exactly as they were during training.

    python scripts/infer_speech.py \
        --ref inference/refs/113_000356120.flac \
        --prompts inference/prompts/dummy_prompts.txt \
        --out inference/out
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def load_model(device: str, adapter_repo: str, base_only: bool):
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device)
    if base_only:
        return model, ta

    from huggingface_hub import snapshot_download
    from peft import PeftModel

    adapter_dir = snapshot_download(adapter_repo, repo_type="model")
    # Wrap the same submodule LoRA training targeted (t3.tfmr), not the
    # top-level model -- attaching it anywhere else silently does nothing.
    model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, adapter_dir)
    model.t3.tfmr.eval()
    print(f"loaded LoRA adapter from {adapter_repo}", flush=True)
    return model, ta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True,
                    help="reference audio for voice/delivery conditioning")
    ap.add_argument("--prompts", type=Path, required=True,
                    help="one prompt per line; blank lines and lines "
                         "starting with # are skipped")
    ap.add_argument("--out", type=Path, default=Path("inference/out"))
    ap.add_argument("--adapter-repo", default="aoxo/text2asmr-chatterbox")
    ap.add_argument("--base-only", action="store_true",
                    help="skip the LoRA adapter, for an A/B baseline")
    ap.add_argument("--exaggeration", type=float, default=0.4,
                    help="lower than Chatterbox's 0.5 default -- ASMR "
                         "delivery is calmer than typical speech")
    ap.add_argument("--cfg-weight", type=float, default=0.3,
                    help="lower than the 0.5 default, which biases toward "
                         "louder/faster delivery than whispered ASMR wants")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    if not args.ref.exists():
        raise SystemExit(f"reference audio not found: {args.ref}")

    import torch

    device = args.device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}", flush=True)

    prompts = [
        line.strip() for line in args.prompts.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not prompts:
        raise SystemExit(f"no prompts found in {args.prompts}")
    print(f"{len(prompts)} prompts", flush=True)

    model, ta = load_model(device, args.adapter_repo, args.base_only)
    model.prepare_conditionals(str(args.ref), exaggeration=args.exaggeration)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "base" if args.base_only else "lora"

    for i, text in enumerate(prompts):
        t0 = time.time()
        wav = model.generate(text, exaggeration=args.exaggeration,
                             cfg_weight=args.cfg_weight)
        dest = args.out / f"{i:02d}_{tag}.wav"
        ta.save(str(dest), wav.cpu(), model.sr)
        dur = wav.shape[-1] / model.sr
        print(f"  [{i+1}/{len(prompts)}] {time.time()-t0:.1f}s to generate "
              f"{dur:.1f}s -> {dest}", flush=True)
        print(f"      {text[:70]!r}", flush=True)

    print(f"\ndone -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
