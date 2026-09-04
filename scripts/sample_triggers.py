#!/usr/bin/env python3
"""Generate a sample from the Stable Audio Open trigger LoRA, locally.

Downloads the gated base model (cached after the first run) and applies the
LoRA checkpoint the same way train_triggers.py's resume patch does: split
across ``model`` and ``conditioner``, since the checkpoint's state_dict has
no matching top-level module of its own (verified empirically during that
fix -- 356/360 tensors land in the model, 4/360 in the conditioner).

    python scripts/sample_triggers.py \
        --checkpoint ckpt/triggers/epoch=0-step=500-v2.ckpt \
        --text "[soft][tapping]" --out sample.wav
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_REPO = "stabilityai/stable-audio-open-1.0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--text", default="[soft][tapping]")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg-scale", type=float, default=6.0)
    ap.add_argument("--out", type=Path, default=Path("trigger_sample.wav"))
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from einops import rearrange
    from huggingface_hub import hf_hub_download
    from stable_audio_tools import create_model_from_config
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}", flush=True)

    print("fetching base model (cached after first run)...", flush=True)
    cfg_path = Path(hf_hub_download(BASE_REPO, "model_config.json"))
    ckpt_path = Path(hf_hub_download(BASE_REPO, "model.ckpt"))
    model_config = json.loads(cfg_path.read_text())

    model = create_model_from_config(model_config)
    base_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_sd = base_ckpt["state_dict"] if "state_dict" in base_ckpt else base_ckpt
    model.load_state_dict(base_sd, strict=False)

    print(f"applying LoRA from {args.checkpoint}...", flush=True)
    lora_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lora_sd = lora_ckpt["state_dict"]
    m1, u1 = model.model.load_state_dict(lora_sd, strict=False)
    m2, u2 = model.conditioner.load_state_dict(lora_sd, strict=False)
    loaded = len(lora_sd) - len(set(u1) & set(u2))
    print(f"resumed LoRA weights: {loaded}/{len(lora_sd)} tensors matched "
          f"(model {len(lora_sd)-len(u1)}, conditioner {len(lora_sd)-len(u2)})")

    model = model.to(device).eval()
    sample_rate = model_config["sample_rate"]
    sample_size = int(args.seconds * sample_rate)

    conditioning = [{
        "prompt": args.text,
        "seconds_start": 0,
        "seconds_total": args.seconds,
    }]

    print(f"sampling {args.seconds}s for text {args.text!r} "
          f"({args.steps} steps, cfg {args.cfg_scale})...", flush=True)
    output = generate_diffusion_cond(
        model,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device,
    )

    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(output.abs().max().clamp(min=1e-8)).clamp(-1, 1)
    sf.write(str(args.out), output.cpu().numpy().T, sample_rate)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
