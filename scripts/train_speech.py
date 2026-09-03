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

import torch


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
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25,
                    help="steps between loss lines; a fraction of total steps "
                         "would put the first line hours into a long run")
    ap.add_argument("--max-consecutive-ooms", type=int, default=40,
                    help="abort if this many batches OOM in a row; a run that "
                         "skips everything looks healthy but learns nothing")
    ap.add_argument("--eval-batches", type=int, default=25,
                    help="batches per eval pass; a sample, not the full split")
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
        # PEFT's save_pretrained writes a README whose base_model frontmatter
        # is empty (get_peft_model was applied to the Llama submodule, not a
        # named pretrained model), which the Hub's metadata validator rejects.
        # The adapter weights don't need the README, so skip it.
        api.upload_folder(folder_path=str(path), repo_id=repo,
                          repo_type="model",
                          ignore_patterns=["README.md"],
                          commit_message=f"checkpoint {path.name}")
        print(f"  pushed {path.name} -> {repo}", flush=True)
    except Exception as exc:  # noqa: BLE001 - backup must not kill training
        print(f"  WARNING: push of {path.name} failed: "
              f"{type(exc).__name__}: {exc}", flush=True)


def load_split(path: Path, smoke: bool):
    """Load training rows from either a packed dataset or the builder's output.

    Packing to parquet embeds the audio, which duplicates ~13 GB on a disk that
    is already tight and takes a slow pass to produce. Training does not need
    it: the builder's ``metadata.jsonl`` plus its FLACs is already a complete,
    directly readable dataset. Packing stays worthwhile for pushing a
    self-contained copy to the Hub, so both layouts are accepted here.
    """
    from datasets import Dataset, load_from_disk

    from datasets import Audio

    meta = path / "metadata.jsonl"
    if meta.exists():
        # Raw builder output. Reuse the packer's reader so the dedup, torn-line
        # and orphan-audio handling is identical in both paths.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.pack_dataset import read_rows, split_by_source

        rows = read_rows(meta)
        train_rows, eval_rows = split_by_source(rows, 0.02)
        for r in rows:
            r["audio"] = str(path / r["file_name"])
        train, evl = Dataset.from_list(train_rows), Dataset.from_list(eval_rows)
    else:
        loaded = load_from_disk(str(path))
        train = loaded["train"] if "train" in loaded else loaded
        evl = loaded["eval"] if "eval" in loaded else None

    train = train.cast_column("audio", Audio(sampling_rate=None))
    if evl is not None and len(evl):
        evl = evl.cast_column("audio", Audio(sampling_rate=None))
    else:
        evl = None

    if smoke:
        train = train.select(range(min(len(train), 16)))
        if evl is not None:
            evl = evl.select(range(min(len(evl), 8)))
    return train, evl


@torch.no_grad()
def evaluate(tuner, loader, max_batches: int) -> float:
    """Mean loss on held-out sources.

    The split is by source recording, so this measures transfer to unseen
    speakers and rooms rather than to unseen segments of a recording the model
    has already heard. A training loss that falls while this does not is
    memorisation, which is the failure worth catching early in a run this long.
    """
    tuner.t3.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss = tuner.loss_only(batch)
        if loss == loss:  # skip nan from an empty batch
            total += loss
            n += 1
    tuner.t3.train()
    return total / max(n, 1)


def main() -> int:
    args = build_argparser().parse_args()
    if args.smoke:
        # Enough steps to exercise forward, backward, optimiser and logging,
        # while staying small enough to run beside other GPU work.
        args.max_steps = args.max_steps or 8
        args.batch = min(args.batch, 2)
        args.grad_accum = 1

    from t2a.models.chatterbox_ft import (
        ChatterboxFinetuner,
        SpeechCollator,
        load_backbone,
    )

    train, evl = load_split(args.data, args.smoke)
    print(f"train examples: {len(train)}"
          f"{f', eval {len(evl)}' if evl is not None else ', no eval split'}",
          flush=True)

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

    eval_loader = (
        torch.utils.data.DataLoader(
            evl, batch_size=args.batch, shuffle=False,
            collate_fn=collator, num_workers=0, drop_last=True,
        ) if evl is not None and len(evl) >= args.batch else None
    )

    steps = args.max_steps or int(len(loader) * args.epochs / args.grad_accum)
    print(f"training for {steps} optimiser steps "
          f"(batch {args.batch} x accum {args.grad_accum})", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)

    # Resume from the newest checkpoint if one exists. The unit restarts on any
    # abnormal exit, and without this every OOM or reboot would discard all
    # progress since the run began.
    step = 0
    if not args.smoke:
        ckpts = sorted(args.out.glob("step-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            step = tuner.resume(ckpts[-1])

    history: list[dict] = []
    hist_path = args.out / "history.json"
    if hist_path.exists():
        try:
            history = json.loads(hist_path.read_text())
        except json.JSONDecodeError:
            history = []
    t0 = time.time()
    start_step = step
    done = step >= steps

    while not done:
        for micro, batch in enumerate(loader):
            loss = tuner.step(batch, accumulate=(micro + 1) % args.grad_accum != 0)
            if tuner.consecutive_ooms >= args.max_consecutive_ooms:
                # Skipping every batch is not "running" -- it burns hours while
                # reporting active and writing checkpoints that never change. Exit
                # non-zero so systemd restarts with a clean allocator and the
                # loop resumes from the last checkpoint.
                print(f"  ABORT: {tuner.consecutive_ooms} consecutive OOMs; "
                      f"exiting so the service restarts with fresh memory",
                      flush=True)
                if step > 0:
                    tuner.save(args.out / f"step-{step}", step=step)
                return 3
            if (micro + 1) % args.grad_accum:
                continue
            step += 1
            if step % args.log_every == 0 or args.smoke:
                rate = (step - start_step) / max(time.time() - t0, 1e-9)
                oom = tuner.oom_skips + collator.oom_skips
                print(f"  step {step}/{steps}  loss {loss:.4f}  "
                      f"{rate:.2f} steps/s"
                      f"{f'  oom_skips {oom}' if oom else ''}", flush=True)
                history.append({"step": step, "loss": loss})
                hist_path.write_text(json.dumps(history, indent=2))
            if eval_loader is not None and step % args.eval_every == 0:
                ev = evaluate(tuner, eval_loader, args.eval_batches)
                print(f"  step {step}  EVAL loss {ev:.4f}  (train {loss:.4f})",
                      flush=True)
                history.append({"step": step, "loss": loss, "eval_loss": ev})
                hist_path.write_text(json.dumps(history, indent=2))
            if step % args.save_every == 0 and not args.smoke:
                ckpt = args.out / f"step-{step}"
                tuner.save(ckpt, step=step)
                if args.push_repo:
                    push_checkpoint(ckpt, args.push_repo, args.push_public)
            if step >= steps:
                done = True
                break

    final = args.out / "final"
    tuner.save(final, step=step)
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    if args.push_repo and not args.smoke:
        push_checkpoint(final, args.push_repo, args.push_public)
    print(f"\ndone in {(time.time() - t0)/60:.1f} min -> {args.out/'final'}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
