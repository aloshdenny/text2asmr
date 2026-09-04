#!/usr/bin/env python3
"""LoRA finetune Stable Audio Open 1.0 on the ASMR trigger set.

Drives stable-audio-tools' own train.py rather than reimplementing the
training loop: unlike Chatterbox, stable-audio-tools ships a working,
documented LoRA path (``training.lora_config`` in the model config), so there
is no equivalent of the T3.loss() bugs to work around here.

What this script actually does:

  1. Fetch model_config.json + model.ckpt from the gated
     stabilityai/stable-audio-open-1.0 repo. This is checked FIRST and fails
     loudly if access has not been granted -- the model page requires
     accepting a license click-through, which does not happen automatically,
     and discovering that after an hour of "training" would be an expensive
     way to find out.
  2. Patch the config: inject lora_config, point learning rate at the LoRA
     range, and write dataset configs referencing the trigger metadata module.
  3. Shell out to train.py with those configs.

Known inefficiency, not hidden: sample_size is left at the shipped value
(the model's native training window). Our trigger clips average 4.4s against
a window built for full songs, so most of every batch is padding. Changing
sample_size safely requires knowing the pretransform's downsampling ratio,
which is only knowable once the gated config is in hand -- doing that blind
risks a shape mismatch that burns pod time to discover. Left as a follow-up
once the first run's real throughput is measured.

    python scripts/train_triggers.py --push-repo aoxo/text2asmr-stable-audio
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE_REPO = "stabilityai/stable-audio-open-1.0"
TOOLS_REPO = "Stability-AI/stable-audio-tools"
WORK = Path("/workspace/sao")


def fetch_train_script(work: Path) -> Path:
    """Get train.py and the defaults.ini it requires.

    `pip install stable-audio-tools` installs the importable package but does
    not register train.py as a console script or module -- it is a repo-root
    script meant to be run from a checkout. Downloading just those two files
    is simpler than cloning the whole tools repo for one script.

    defaults.ini is not optional: prefigure's ``get_all_args()`` reads it via
    a relative path from the process's working directory, and fails with an
    opaque ``configparser.NoSectionError`` -> ``TypeError`` (a bug in
    prefigure's own error handling swallows the real message) if it's
    missing. Downloading it alongside train.py and running from this
    directory keeps that relative path valid regardless of where the caller's
    own CWD happens to be.
    """
    import urllib.request

    for name in ("train.py", "defaults.ini"):
        dest = work / name
        url = f"https://raw.githubusercontent.com/{TOOLS_REPO}/main/{name}"
        urllib.request.urlretrieve(url, dest)
        if dest.stat().st_size < 20:
            raise SystemExit(f"{name} fetch from {url} looks truncated")
    return work / "train.py"


def require_gated_access() -> tuple[Path, Path]:
    """Download the base model, failing fast and legibly if access is gated.

    A 403 here means the pod's HF account has not accepted the license on
    huggingface.co/stabilityai/stable-audio-open-1.0. That is a five-second
    manual step for a human and an opaque stack trace for anyone reading pod
    logs later, so it is caught and re-raised with the actual fix.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        cfg = Path(hf_hub_download(BASE_REPO, "model_config.json"))
        ckpt = Path(hf_hub_download(BASE_REPO, "model.ckpt"))
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise SystemExit(
            f"Cannot access {BASE_REPO}: {exc}\n\n"
            f"This repo is gated. Visit https://huggingface.co/{BASE_REPO} "
            f"and accept the license with the account whose token is set as "
            f"HF_TOKEN, then re-run."
        ) from exc
    return cfg, ckpt


def build_configs(cfg_path: Path, out_dir: Path, lora_rank: int, lora_alpha: int,
                  lr: float, triggers_dir: Path, metadata_module: Path) -> tuple[Path, Path]:
    model_config = json.loads(cfg_path.read_text())
    model_config.setdefault("training", {})
    model_config["training"]["learning_rate"] = lr
    model_config["training"]["lora_config"] = {
        "rank": lora_rank,
        "alpha": lora_alpha,
        "adapter_type": "lora",
    }
    out_model = out_dir / "model_config.json"
    out_model.write_text(json.dumps(model_config, indent=2))

    dataset_config = {
        "dataset_type": "audio_dir",
        "datasets": [{
            "id": "asmr_triggers",
            "path": str(triggers_dir),
            "custom_metadata_module": str(metadata_module),
        }],
        "random_crop": True,
    }
    out_dataset = out_dir / "dataset_config.json"
    out_dataset.write_text(json.dumps(dataset_config, indent=2))
    return out_model, out_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triggers-dir", type=Path,
                    default=Path("/workspace/out/triggers"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/ckpt/triggers"))
    ap.add_argument("--metadata-module", type=Path,
                    default=Path("/workspace/text2asmr/t2a/data/trigger_metadata.py"))
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum-batches", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="low by design: a worker gets SIGKILLed by the "
                         "pod's cgroup after tens of minutes regardless of "
                         "num_workers (looks like an upstream memory leak in "
                         "stable-audio-tools, not something we can fix here), "
                         "so a restart should lose only a couple of minutes")
    ap.add_argument("--val-every", type=int, default=-1)
    ap.add_argument("--max-retries", type=int, default=20,
                    help="training resumes from the latest checkpoint after "
                         "each crash, up to this many attempts")
    ap.add_argument("--push-repo", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="DOES NOT LIMIT STEP COUNT. train.py's --max-steps "
                         "CLI arg is parsed but never passed to pl.Trainer(), "
                         "so it is a silent no-op -- confirmed by grepping "
                         "the actual train.py for the string, and by a "
                         "'smoke' run that trained 677 real steps over 12 "
                         "minutes before crashing. This flag only lowers "
                         "batch size, as a mild memory-pressure reduction; "
                         "it does not make a run short or cheap.")
    args = ap.parse_args()

    if not args.metadata_module.exists():
        raise SystemExit(f"metadata module not found: {args.metadata_module}")
    n_flac = len(list(args.triggers_dir.glob("*.flac")))
    if n_flac < 15000:
        raise SystemExit(f"only {n_flac} trigger clips at {args.triggers_dir}; "
                         f"expected ~20,340 -- run fetch_triggers.py first")
    print(f"{n_flac:,} trigger clips at {args.triggers_dir}", flush=True)

    print(f"checking access to {BASE_REPO}...", flush=True)
    cfg_path, ckpt_path = require_gated_access()
    print("access OK", flush=True)

    WORK.mkdir(parents=True, exist_ok=True)
    model_cfg, dataset_cfg = build_configs(
        cfg_path, WORK, args.lora_rank, args.lora_alpha, args.lr,
        args.triggers_dir, args.metadata_module,
    )

    train_py = fetch_train_script(WORK)
    args.out.mkdir(parents=True, exist_ok=True)

    def base_cmd() -> list[str]:
        return [
            sys.executable, str(train_py),
            "--model-config", str(model_cfg),
            "--dataset-config", str(dataset_cfg),
            "--pretrained-ckpt-path", str(ckpt_path),
            "--name", "t2a-triggers-smoke" if args.smoke else "t2a-triggers",
            "--save-dir", str(args.out),
            "--batch-size", str(2 if args.smoke else args.batch_size),
            "--accum-batches", str(1 if args.smoke else args.accum_batches),
            "--checkpoint-every", str(args.checkpoint_every),
            "--val-every", str(args.val_every),
            "--logger", "none",
            # No --num-gpus: not a real train.py flag (rejected with
            # "unrecognized arguments") despite the source referencing
            # args.num_gpus -- prefigure derives that internally.
            "--num-nodes", "1",
            # 1 is the practical floor: 0 is rejected outright because
            # stable-audio-tools hardcodes persistent_workers=True, which
            # requires num_workers > 0. Lower counts did not stop the SIGKILL
            # (2 died further into training than 6 did, 1 further still, all
            # after real progress -- 677 steps at num_workers=1 before dying)
            # so this looks like a genuine upstream memory leak, not a
            # worker-count problem. Handled below via checkpoint + resume.
            "--num-workers", "1",
            "--precision", "16-mixed",
        ]

    def latest_checkpoint() -> Path | None:
        ckpts = sorted(args.out.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        return ckpts[-1] if ckpts else None

    for attempt in range(1, args.max_retries + 1):
        cmd = base_cmd()
        resume = latest_checkpoint()
        if resume is not None:
            # Full Lightning trainer state (optimizer, step count), not just
            # weights -- --pretrained-ckpt-path above stays fixed as the base
            # model; --ckpt-path is what actually continues a run.
            cmd += ["--ckpt-path", str(resume)]
            print(f"attempt {attempt}: resuming from {resume}", flush=True)
        else:
            print(f"attempt {attempt}: starting fresh", flush=True)

        print("running:", " ".join(cmd), flush=True)
        # cwd=WORK so prefigure finds defaults.ini via its relative-path
        # lookup, regardless of where this script itself was invoked from.
        result = subprocess.run(cmd, cwd=WORK)
        if result.returncode == 0:
            break
        print(f"attempt {attempt} exited {result.returncode}", flush=True)
        if args.smoke:
            # A crash on the very first attempt with no checkpoint yet means
            # something is wrong with the config/data, not the leak -- don't
            # burn retries on that.
            if latest_checkpoint() is None:
                return result.returncode
    else:
        print(f"gave up after {args.max_retries} attempts", flush=True)
        return 1

    if args.smoke:
        print("smoke passed (reached a real checkpoint)", flush=True)
        return 0

    if args.push_repo:
        push_checkpoints(args.out, args.push_repo)
    return 0


def push_checkpoints(out_dir: Path, repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    ckpts = sorted(out_dir.rglob("*.ckpt"))
    if not ckpts:
        print(f"WARNING: no checkpoints found under {out_dir}", flush=True)
        return
    latest = ckpts[-1]
    try:
        api.upload_file(path_or_fileobj=str(latest), path_in_repo=latest.name,
                        repo_id=repo, repo_type="model",
                        commit_message=f"checkpoint {latest.name}")
        print(f"pushed {latest.name} -> {repo}", flush=True)
    except Exception as exc:  # noqa: BLE001 - a failed push must not fail the run
        print(f"WARNING: push failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
