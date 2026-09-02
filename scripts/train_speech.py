#!/usr/bin/env python3
"""Finetune Chatterbox's T3 text-to-speech-token model on ASMR speech.

Chatterbox is three pieces: a Llama-backed T3 that maps text to S3 speech
tokens, an S3Gen vocoder that turns those tokens into a waveform, and a voice
encoder for the reference-audio conditioning. Adapting it to whispered ASMR
delivery is a T3 problem -- the prosody, breathiness and pacing all live in the
token sequence -- so S3Gen and the voice encoder stay frozen. That also keeps
the job inside a 20 GB card shared with other work.

Training is a causal-LM objective over the speech tokens, conditioned on the
text and on a reference clip drawn from the same source recording, so the model
learns the delivery rather than memorising one speaker.

    # smoke test: a handful of steps, proves the loop runs
    python scripts/train_speech.py --data ~/t2a/packed/speech --smoke

    # real run
    python scripts/train_speech.py --data ~/t2a/packed/speech \
        --epochs 2 --batch 4 --out ~/t2a/ckpt/speech
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="dataset saved by pack_dataset.py")
    ap.add_argument("--out", type=Path, default=Path("ckpt/speech"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = derive from epochs")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="run a few steps on a tiny slice and exit")
    ap.add_argument("--push-repo", default="",
                    help="hub model repo to mirror checkpoints to as they are "
                         "written; requires `hf auth login` on this machine")
    ap.add_argument("--push-public", action="store_true",
                    help="create the checkpoint repo public (default private)")
    return ap


def push_checkpoint(path: Path, repo: str, public: bool) -> None:
    """Mirror a checkpoint to the Hub, never failing the training run.

    The box this trains on has died mid-job more than once, so checkpoints are
    pushed as they are written rather than at the end. A failed push must not
    take the run with it -- losing a backup is recoverable, losing the run is
    not.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, repo_type="model", private=not public,
                        exist_ok=True)
        api.upload_folder(folder_path=str(path), repo_id=repo,
                          repo_type="model",
                          commit_message=f"checkpoint {path.name}")
        print(f"  pushed {path.name} -> {repo}", flush=True)
    except Exception as exc:  # noqa: BLE001 - backup must not kill training
        print(f"  WARNING: push of {path.name} failed: "
              f"{type(exc).__name__}: {exc}", flush=True)


def main() -> int:
    args = build_argparser().parse_args()
    if args.smoke:
        # Enough steps to exercise forward, backward, optimiser and logging,
        # while staying small enough to run beside other GPU work.
        args.max_steps = args.max_steps or 8
        args.batch = min(args.batch, 2)
        args.grad_accum = 1

    import torch
    from datasets import load_from_disk

    from t2a.models.chatterbox_ft import (
        ChatterboxFinetuner,
        SpeechCollator,
        load_backbone,
    )

    ds = load_from_disk(str(args.data))
    train = ds["train"] if "train" in ds else ds
    if args.smoke:
        train = train.select(range(min(len(train), 16)))
    print(f"train examples: {len(train)}", flush=True)

    backbone = load_backbone(args.device)
    tuner = ChatterboxFinetuner(
        backbone,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        warmup=args.warmup,
    )
    collator = SpeechCollator(backbone)

    loader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collator,
        # Must stay 0: the collator holds the Chatterbox model and runs the S3
        # tokenizer and voice encoder on the GPU. Worker processes would have
        # to pickle it, which fails on s3gen's parametrized conv layers, and
        # would be wrong anyway -- that work belongs in the main process.
        num_workers=0,
        drop_last=True,
    )

    steps = args.max_steps or int(len(loader) * args.epochs / args.grad_accum)
    print(f"training for {steps} optimiser steps "
          f"(batch {args.batch} x accum {args.grad_accum})", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    t0 = time.time()
    step = 0
    done = False

    while not done:
        for micro, batch in enumerate(loader):
            loss = tuner.step(batch, accumulate=(micro + 1) % args.grad_accum != 0)
            if (micro + 1) % args.grad_accum:
                continue
            step += 1
            if step % max(steps // 20, 1) == 0 or args.smoke:
                rate = step / max(time.time() - t0, 1e-9)
                print(f"  step {step}/{steps}  loss {loss:.4f}  "
                      f"{rate:.2f} steps/s", flush=True)
                history.append({"step": step, "loss": loss})
            if step % args.save_every == 0 and not args.smoke:
                ckpt = args.out / f"step-{step}"
                tuner.save(ckpt)
                if args.push_repo:
                    push_checkpoint(ckpt, args.push_repo, args.push_public)
            if step >= steps:
                done = True
                break

    final = args.out / "final"
    tuner.save(final)
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    if args.push_repo and not args.smoke:
        push_checkpoint(final, args.push_repo, args.push_public)
    print(f"\ndone in {(time.time() - t0)/60:.1f} min -> {args.out/'final'}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
