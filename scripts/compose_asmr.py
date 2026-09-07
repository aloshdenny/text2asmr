#!/usr/bin/env python3
"""Render one bracket-tag script into a single mixed speech+trigger clip.

Neither finetuned model can produce this alone: Chatterbox only generates
speech, Stable Audio Open only generates triggers (see text2asmr.compose.
grammar, which parses the paper's bracket-tag notation into an ordered
Script but stops short of rendering audio). This is the missing renderer:
it walks the parsed segments in order, generates each with the model that
owns it, resamples everything to one common rate, and concatenates them
with a short crossfade so segment boundaries don't click.

Chatterbox (t2a-infer env) and Stable Audio Open (sao-infer env) are
deliberately separate conda environments -- installing both models' deps
into one env risks exactly the kind of conflict that separating them was
meant to avoid. So this runs in three --stage invocations, each in its own
env, handing off through a shared work directory of per-segment .npy files
identified by their index in the script (not id(), which is only valid
within a single process and wouldn't survive the handoff) rather than
trying to import both models in one process:

    T2A_INFER=/opt/homebrew/Caskroom/miniconda/base/envs/t2a-infer/bin/python3
    SAO_INFER=/opt/homebrew/Caskroom/miniconda/base/envs/sao-infer/bin/python3
    WORK=inference/out/compose_work

    $T2A_INFER scripts/compose_asmr.py --stage speech --work-dir $WORK \
        --ref inference/refs/113_000356120.flac
    $SAO_INFER scripts/compose_asmr.py --stage trigger --work-dir $WORK \
        --trigger-checkpoint ckpt/triggers/epoch=0-step=2600.ckpt
    $T2A_INFER scripts/compose_asmr.py --stage stitch --work-dir $WORK \
        --out inference/out/composed_demo.wav

Every stage must be passed the same --script (or rely on the shared
DEFAULT_SCRIPT below) so parse() -- deterministic and pure -- produces the
same segment order and count each time, without needing a separate
manifest file to agree on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# A representative ASMR script in the paper's own grammar: speech spans
# interleaved with intensity+trigger tags. Sized to land close to Stable
# Audio Open's native 47.55s ceiling once rendered, not an exact fit --
# Chatterbox's own output length for a given line of text isn't something
# this script controls directly.
DEFAULT_SCRIPT = (
    "Hey there, welcome back. I'm so glad you decided to spend some time "
    "with me tonight. [soft][tapping] Just relax, and let your shoulders "
    "drop for me. [mild][brushing] There's no rush at all, we have all the "
    "time you need. [soft][crinkling] Doesn't that feel nice, right there "
    "by your ear? [vigorous][tapping] Good. Just let everything else fade "
    "away for a little while. [soft][breathing close] That's it, just "
    "breathe with me. [mild][fabric rustling] You're doing so well. Let's "
    "keep going a little longer."
)

BASE_REPO = "stabilityai/stable-audio-open-1.0"


def stage_speech(script, work_dir: Path, ref: Path, adapter_repo: str,
                 exaggeration: float, cfg_weight: float, device: str) -> None:
    import numpy as np
    from chatterbox.tts import ChatterboxTTS
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    print(f"[speech] loading Chatterbox + LoRA adapter ({adapter_repo})...", flush=True)
    model = ChatterboxTTS.from_pretrained(device)
    adapter_dir = snapshot_download(adapter_repo, repo_type="model")
    model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, adapter_dir)
    model.t3.tfmr.eval()
    model.prepare_conditionals(str(ref), exaggeration=exaggeration)
    print("[speech] model loaded", flush=True)

    for i, seg in enumerate(script.segments):
        if seg.kind != "speech":
            continue
        wav = model.generate(seg.text, exaggeration=exaggeration, cfg_weight=cfg_weight)
        arr = wav.squeeze(0).cpu().numpy()
        dur = len(arr) / model.sr
        np.save(work_dir / f"seg_{i:03d}.npy", arr)
        (work_dir / f"seg_{i:03d}.sr").write_text(str(model.sr))
        print(f"  [speech] seg {i}: {dur:.1f}s: {seg.text[:60]!r}", flush=True)


def stage_trigger(script, work_dir: Path, checkpoint: Path, seconds: float,
                  steps: int, cfg_scale: float, device: str) -> None:
    import json
    import numpy as np
    import torch
    from einops import rearrange
    from huggingface_hub import hf_hub_download
    from stable_audio_tools import create_model_from_config
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    print("[trigger] loading Stable Audio Open + LoRA...", flush=True)
    cfg_path = Path(hf_hub_download(BASE_REPO, "model_config.json"))
    ckpt_path = Path(hf_hub_download(BASE_REPO, "model.ckpt"))
    model_config = json.loads(cfg_path.read_text())

    model = create_model_from_config(model_config)
    base_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_sd = base_ckpt["state_dict"] if "state_dict" in base_ckpt else base_ckpt
    model.load_state_dict(base_sd, strict=False)

    lora_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lora_sd = lora_ckpt["state_dict"]
    model.model.load_state_dict(lora_sd, strict=False)
    model.conditioner.load_state_dict(lora_sd, strict=False)
    model = model.to(device).eval()
    sr = model_config["sample_rate"]
    print("[trigger] model loaded", flush=True)

    sample_size = int(seconds * sr)
    for i, seg in enumerate(script.segments):
        if seg.kind != "trigger":
            continue
        conditioning = [{"prompt": seg.prompt, "seconds_start": 0, "seconds_total": seconds}]
        output = generate_diffusion_cond(
            model, steps=steps, cfg_scale=cfg_scale, conditioning=conditioning,
            sample_size=sample_size, sigma_min=0.3, sigma_max=500,
            sampler_type="dpmpp-3m-sde", device=device,
        )
        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(output.abs().max().clamp(min=1e-8)).clamp(-1, 1)
        arr = output.mean(dim=0).cpu().numpy()  # stereo -> mono, matches speech
        np.save(work_dir / f"seg_{i:03d}.npy", arr)
        (work_dir / f"seg_{i:03d}.sr").write_text(str(sr))
        print(f"  [trigger] seg {i}: {seconds:.1f}s: {seg.prompt!r}", flush=True)


def stage_stitch(script, work_dir: Path, out: Path, target_sr: int,
                 crossfade_s: float) -> None:
    import librosa
    import numpy as np
    import soundfile as sf

    pieces = []
    missing = []
    for i, seg in enumerate(script.segments):
        npy_path = work_dir / f"seg_{i:03d}.npy"
        sr_path = work_dir / f"seg_{i:03d}.sr"
        if not npy_path.exists():
            missing.append((i, seg.kind))
            continue
        wav = np.load(npy_path)
        sr = int(sr_path.read_text())
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        pieces.append(wav)

    if missing:
        raise SystemExit(f"missing rendered segments (run the earlier stages "
                         f"first): {missing}")
    if not pieces:
        raise SystemExit("nothing to stitch -- script produced no renderable segments")

    fade_n = int(crossfade_s * target_sr)
    out_wav = pieces[0]
    for nxt in pieces[1:]:
        if fade_n > 0 and len(out_wav) >= fade_n and len(nxt) >= fade_n:
            fade_out = np.linspace(1, 0, fade_n)
            fade_in = np.linspace(0, 1, fade_n)
            head = out_wav[:-fade_n]
            tail = out_wav[-fade_n:] * fade_out + nxt[:fade_n] * fade_in
            out_wav = np.concatenate([head, tail, nxt[fade_n:]])
        else:
            out_wav = np.concatenate([out_wav, nxt])

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), out_wav, target_sr)
    dur = len(out_wav) / target_sr
    print(f"\n-> {out} ({dur:.1f}s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["speech", "trigger", "stitch"])
    ap.add_argument("--script", default=DEFAULT_SCRIPT)
    ap.add_argument("--work-dir", type=Path, default=Path("inference/out/compose_work"))
    ap.add_argument("--ref", type=Path, help="required for --stage speech")
    ap.add_argument("--speech-adapter", default="aoxo/text2asmr-chatterbox")
    ap.add_argument("--trigger-checkpoint", type=Path, help="required for --stage trigger")
    ap.add_argument("--trigger-seconds", type=float, default=4.0)
    ap.add_argument("--trigger-steps", type=int, default=100)
    ap.add_argument("--trigger-cfg-scale", type=float, default=6.0)
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--crossfade-s", type=float, default=0.15)
    ap.add_argument("--target-sr", type=int, default=44100,
                    help="Stable Audio Open's native rate, the higher of "
                        "the two models' -- everything upsamples to it "
                        "rather than losing resolution from the triggers")
    ap.add_argument("--out", type=Path, default=Path("inference/out/composed.wav"))
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    from text2asmr.compose.grammar import parse

    script = parse(args.script)
    print(f"parsed {len(script.speech)} speech span(s), "
          f"{len(script.triggers)} trigger(s)", flush=True)

    args.work_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = args.device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    if args.stage == "speech":
        if not args.ref:
            raise SystemExit("--ref is required for --stage speech")
        print(f"device: {device}", flush=True)
        stage_speech(script, args.work_dir, args.ref, args.speech_adapter,
                    args.exaggeration, args.cfg_weight, device)
    elif args.stage == "trigger":
        if not args.trigger_checkpoint:
            raise SystemExit("--trigger-checkpoint is required for --stage trigger")
        print(f"device: {device}", flush=True)
        stage_trigger(script, args.work_dir, args.trigger_checkpoint,
                     args.trigger_seconds, args.trigger_steps,
                     args.trigger_cfg_scale, device)
    else:
        stage_stitch(script, args.work_dir, args.out, args.target_sr, args.crossfade_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
