#!/usr/bin/env python3
"""Generate a sample from a native T2A checkpoint, to check it against loss alone.

Loss going down is not proof the model is good -- it could be exploiting
zero-padded short clips, or diverged in a way MSE doesn't penalize much. This
actually runs the DDPM sampling loop and writes real audio, which loss numbers
can't fake.

    python scripts/sample_native.py --checkpoint ~/text2asmr/ckpt/native/step-13500.pt \
        --text "[soft][tapping]" --out sample.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--text", default="[soft][tapping]")
    ap.add_argument("--out", type=Path, default=Path("native_sample.wav"))
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from transformers import BertTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from text2asmr.native.dataset import SR
    from text2asmr.native.diffusion import DDPM
    from text2asmr.native.model import T2ANative

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}", flush=True)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = T2ANative(cond_dim=state["cond_dim"], channels=tuple(state["channels"]))
    model.load_state_dict(state["model"], strict=False)  # BERT reloads separately
    model.to(device).eval()
    print(f"loaded checkpoint from step {state['step']}", flush=True)

    ddpm = DDPM(timesteps=1000, device=device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    enc = tokenizer(args.text, truncation=True, max_length=32,
                    padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    window = int(state["window_s"] * SR)
    window -= window % model.length_multiple
    print(f"sampling {window/SR:.2f}s for text {args.text!r} "
          f"(this runs 1000 sequential denoising steps, expect it to be slow)",
          flush=True)

    sample = ddpm.sample(model, (1, 1, window), input_ids, attention_mask, device)
    audio = sample.squeeze().detach().cpu().numpy()

    print(f"stats: min={audio.min():.3f} max={audio.max():.3f} "
          f"std={audio.std():.4f} (near-zero std means degenerate output)")

    sf.write(args.out, audio, SR)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
