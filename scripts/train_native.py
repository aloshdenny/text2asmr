#!/usr/bin/env python3
"""Train the native T2A architecture: BERT -> FiLM U-Net -> DDPM, from scratch.

Companion to the Stable Audio Open / Chatterbox finetunes, built to compare
the paper's own architecture against finetuned pretrained backbones on the
same corpus.

Checkpointing and resume are built in from the start this time, not bolted on
after a crash -- every prior training job in this project has needed it
eventually (the speech run restarted repeatedly across box reboots and a
watchdog kill; the trigger run needed it after a genuine upstream leak). A
plain ``torch.save``/``torch.load`` round-trip is enough here, since this is
our own training loop, not a third-party script whose checkpoint format has
to be reverse-engineered.

    python scripts/train_native.py \
        --speech-dir ~/text2asmr/out/speech --triggers-dir ~/text2asmr/out/triggers \
        --out ~/text2asmr/ckpt/native --steps 20000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-dir", type=Path, default=None)
    ap.add_argument("--triggers-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window-s", type=float, default=3.0,
                    help="fixed audio window length; short by design so a "
                         "modest U-Net + DDPM trains at a reasonable speed")
    ap.add_argument("--channels", default="32,64,128,256,512",
                    help="U-Net channel widths, comma-separated")
    ap.add_argument("--cond-dim", type=int, default=256)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    if args.speech_dir is None and args.triggers_dir is None:
        raise SystemExit("need at least one of --speech-dir / --triggers-dir")

    import torch
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from text2asmr.native.dataset import T2ANativeDataset
    from text2asmr.native.diffusion import DDPM
    from text2asmr.native.model import T2ANative

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}", flush=True)

    channels = tuple(int(c) for c in args.channels.split(","))
    model = T2ANative(cond_dim=args.cond_dim, channels=channels).to(device)
    ddpm = DDPM(timesteps=args.timesteps, device=device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable/1e6:.2f}M "
          f"(paper's own checkpoint is ~3M)", flush=True)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = T2ANativeDataset(args.speech_dir, args.triggers_dir,
                               args.window_s, model.length_multiple, tokenizer)
    print(f"dataset: {len(dataset)} items", flush=True)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)
    hist_path = args.out / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []

    # Resume: load the latest checkpoint's model/optimizer/step if one exists.
    # A plain torch.save/load pair works here specifically because this is
    # our own model and our own training loop -- no third-party checkpoint
    # format to reverse-engineer, unlike the Stable Audio Open LoRA resume.
    ckpts = sorted(args.out.glob("step-*.pt"),
                   key=lambda p: int(p.stem.split("-")[1]))
    start_step = 0
    if ckpts:
        state = torch.load(ckpts[-1], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step = state["step"]
        print(f"resumed from {ckpts[-1]} at step {start_step}", flush=True)

    def save(step: int) -> None:
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "channels": channels,
            "cond_dim": args.cond_dim,
            "window_s": args.window_s,
        }, args.out / f"step-{step}.pt")
        print(f"  saved checkpoint at step {step}", flush=True)

    model.train()
    step = start_step
    t0 = time.time()
    loader_iter = iter(loader)

    while step < args.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        waveform = batch["waveform"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        opt.zero_grad(set_to_none=True)
        loss = ddpm.loss(model, waveform, input_ids, attention_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % args.log_every == 0:
            rate = (step - start_step) / max(time.time() - t0, 1e-9)
            entry = {"step": step, "loss": float(loss.detach())}
            history.append(entry)
            hist_path.write_text(json.dumps(history, indent=2))
            print(f"  step {step}/{args.steps}  loss {entry['loss']:.4f}  "
                  f"{rate:.2f} steps/s", flush=True)

        if step % args.save_every == 0:
            save(step)

    save(step)
    print(f"\ndone at step {step} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
