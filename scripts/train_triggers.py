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

    python scripts/train_triggers.py --smoke
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
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=-1)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--push-repo", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="a handful of steps, to prove the config and dataset "
                         "load correctly before a run that bills for hours")
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
    cmd = [
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
        "--max-steps", str(10 if args.smoke else args.max_steps),
        "--logger", "none",
        # No --num-gpus: it isn't a real train.py flag (rejected with
        # "unrecognized arguments") despite train.py's own source referencing
        # args.num_gpus -- that value comes from somewhere prefigure derives
        # internally, not from the CLI. Single GPU needs no override anyway.
        "--num-nodes", "1",
        # Default is 6. The pod's cgroup caps the container at ~58GB despite
        # the host having 503GB, and a DataLoader worker was SIGKILLed --
        # almost certainly the same corrupt-header file that broke
        # retag_triggers.py's allocator. Fewer workers means less simultaneous
        # decode/prefetch memory in flight if it (or another bad file) is hit
        # again before it's identified and removed from the corpus.
        "--num-workers", "0",
        "--precision", "16-mixed",
    ]
    print("running:", " ".join(cmd), flush=True)
    # cwd=WORK so prefigure finds defaults.ini via its relative-path lookup,
    # regardless of where this script itself was invoked from.
    result = subprocess.run(cmd, cwd=WORK)
    if result.returncode != 0:
        return result.returncode

    if args.smoke:
        print("smoke test passed", flush=True)
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
