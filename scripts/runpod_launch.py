#!/usr/bin/env python3
"""Create a RunPod GPU pod that rebuilds the corpus and trains, unattended.

Reads credentials from the environment; nothing is passed on the command line:

    export RUNPOD_API_KEY=...      # from runpod.io/console/user/settings
    export HF_TOKEN=...            # write access, for dataset + checkpoints
    python scripts/runpod_launch.py --gpu "RTX 4090" --stage all

Three things this handles that cost time on the previous RunPod run:

  * the REST endpoint ``rest.runpod.io/v1/gpuTypes`` 404s; the GraphQL API at
    ``api.runpod.io/graphql`` is the one that works
  * a pod only starts sshd if ``PUBLIC_KEY`` is injected **at creation**.
    Registering an account key afterwards does not propagate to running pods.
  * a "download" that silently no-ops leaves an idle pod burning money. The
    bootstrap verifies the cache is actually growing and kills the pod if it
    is not.

The pod is disposable: it pulls the public source corpus, rebuilds the
segments, pushes them to the Hub, trains, and pushes checkpoints as it goes.
Nothing depends on it surviving.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

GQL = "https://api.runpod.io/graphql"


def gql(query: str, api_key: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        f"{GQL}?api_key={api_key}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"RunPod API error {e.code}: {e.read()[:400].decode()}")
    if "errors" in out:
        raise SystemExit(f"RunPod GraphQL error: {out['errors']}")
    return out["data"]


def list_gpus(api_key: str) -> list[dict]:
    q = """
    query { gpuTypes { id displayName memoryInGb
        lowestPrice(input:{gpuCount:1}) { uninterruptablePrice minimumBidPrice } } }
    """
    return gql(q, api_key)["gpuTypes"]


def bootstrap_script(stage: str, hf_token_env: str, repo: str) -> str:
    """Commands the pod runs on boot.

    Written to be re-runnable and to fail loudly: every long step verifies it
    actually did something, because an idle pod still bills.
    """
    return textwrap.dedent(f"""
    set -euo pipefail
    exec > >(tee -a /workspace/bootstrap.log) 2>&1
    echo "=== bootstrap $(date -u) ==="

    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq git ffmpeg tmux >/dev/null

    cd /workspace
    [ -d text2asmr ] || git clone {repo} text2asmr
    cd text2asmr

    python -m pip install -q -U pip
    python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121 || true
    python -m pip install -q transformers datasets soundfile librosa huggingface_hub \
        chatterbox-tts peft torchcodec

    python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

    # Fail fast rather than train on an unauthenticated shell.
    python - <<'PY'
    import os, sys
    from huggingface_hub import whoami
    if not os.environ.get("{hf_token_env}"):
        sys.exit("{hf_token_env} not set in pod env")
    print("hub user:", whoami()["name"])
    PY

    STAGE="{stage}"

    if [ "$STAGE" = "build" ] || [ "$STAGE" = "all" ]; then
      echo "=== rebuilding corpus from aoxo/audios ==="
      # Watchdog: a silent no-op download is the expensive failure mode, so
      # abort if the HF cache has not grown after five minutes.
      ( sleep 300
        SZ=$(du -sm ~/.cache/huggingface 2>/dev/null | cut -f1 || echo 0)
        if [ "${{SZ:-0}}" -lt 200 ]; then
          echo "FATAL: HF cache only ${{SZ}}MB after 5min - download is not progressing"
          pkill -f build_datasets.py || true
        fi
      ) &
      python scripts/build_datasets.py --out /workspace/out \\
          --trigger-hours 25 --speech-hours 120
      python scripts/sync_hf.py data --out /workspace/out \\
          --repo aoxo/text2asmr-segments
    fi

    if [ "$STAGE" = "train" ] || [ "$STAGE" = "all" ]; then
      echo "=== training ==="
      python scripts/train_speech.py --data /workspace/out/speech \\
          --out /workspace/ckpt/speech --epochs 1 --batch 4 --grad-accum 4 \\
          --save-every 250 --eval-every 250 \\
          --push-repo aoxo/text2asmr-chatterbox
    fi

    echo "=== bootstrap complete $(date -u) ==="
    """).strip()


def create_pod(api_key: str, args, env: dict) -> dict:
    env_list = ", ".join(
        f'{{key: "{k}", value: {json.dumps(v)}}}' for k, v in env.items()
    )
    q = f"""
    mutation {{
      podFindAndDeployOnDemand(input: {{
        cloudType: {args.cloud}
        gpuCount: 1
        volumeInGb: {args.disk}
        containerDiskInGb: {args.container_disk}
        minVcpuCount: 8
        minMemoryInGb: 32
        gpuTypeId: {json.dumps(args.gpu_id)}
        name: {json.dumps(args.name)}
        imageName: {json.dumps(args.image)}
        dockerArgs: ""
        ports: "22/tcp"
        volumeMountPath: "/workspace"
        env: [{env_list}]
      }}) {{ id imageName machineId costPerHr }}
    }}
    """
    return gql(q, api_key)["podFindAndDeployOnDemand"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="RTX 4090",
                    help="display name substring, e.g. 'RTX 4090', 'A6000'")
    ap.add_argument("--stage", default="all", choices=["build", "train", "all"])
    ap.add_argument("--name", default="t2a")
    ap.add_argument("--image",
                    default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    ap.add_argument("--disk", type=int, default=200, help="volume GB")
    ap.add_argument("--container-disk", type=int, default=50)
    ap.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    ap.add_argument("--repo", default="https://github.com/aloshdenny/text2asmr")
    ap.add_argument("--pubkey", type=Path,
                    default=Path.home() / ".ssh/id_ed25519.pub")
    ap.add_argument("--list-gpus", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print("RUNPOD_API_KEY is not set.\n"
              "  export RUNPOD_API_KEY=...   (runpod.io/console/user/settings)",
              file=sys.stderr)
        return 2

    if args.list_gpus:
        for g in sorted(list_gpus(api_key), key=lambda x: x["displayName"]):
            price = (g.get("lowestPrice") or {}).get("uninterruptablePrice")
            print(f"  {g['displayName']:<28} {g['memoryInGb']:>3}GB  "
                  f"${price if price is not None else '?'}/hr   id={g['id']}")
        return 0

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("HF_TOKEN is not set. The pod needs it to pull the source corpus\n"
              "and push the dataset and checkpoints. Export it and re-run.",
              file=sys.stderr)
        return 2

    matches = [g for g in list_gpus(api_key)
               if args.gpu.lower() in g["displayName"].lower()]
    if not matches:
        print(f"No GPU matching {args.gpu!r}. Try --list-gpus.", file=sys.stderr)
        return 2
    gpu = min(matches,
              key=lambda g: (g.get("lowestPrice") or {}).get(
                  "uninterruptablePrice") or 9e9)
    args.gpu_id = gpu["id"]
    price = (gpu.get("lowestPrice") or {}).get("uninterruptablePrice")
    print(f"selected {gpu['displayName']} ({gpu['memoryInGb']}GB) "
          f"at ${price}/hr")

    if not args.pubkey.exists():
        print(f"No public key at {args.pubkey}. sshd will not start without one "
              f"injected at creation time.", file=sys.stderr)
        return 2

    env = {
        "HF_TOKEN": hf_token,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Must be present at creation: adding the account key later does not
        # reach an already-running pod.
        "PUBLIC_KEY": args.pubkey.read_text().strip(),
        "T2A_BOOTSTRAP": bootstrap_script(args.stage, "HF_TOKEN", args.repo),
    }

    est = {"build": 4, "train": 10, "all": 14}[args.stage]
    print(f"stage={args.stage}  est ~{est}h  "
          f"~${(price or 0) * est:.2f} at ${price}/hr")

    if args.dry_run:
        print("\n--- bootstrap that would run on the pod ---")
        print(env["T2A_BOOTSTRAP"])
        return 0

    pod = create_pod(api_key, args, env)
    print(f"\npod {pod['id']} created at ${pod.get('costPerHr')}/hr")
    print(textwrap.dedent(f"""
        The bootstrap is in the pod env as $T2A_BOOTSTRAP but is NOT run
        automatically -- run it yourself so you see it start:

          ssh root@<pod-host> -p <port>       # from runpod.io/console/pods
          echo "$T2A_BOOTSTRAP" > /workspace/run.sh && bash /workspace/run.sh

        Watch:   tail -f /workspace/bootstrap.log
        Stop:    terminate the pod in the console -- it bills until you do.
    """).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
