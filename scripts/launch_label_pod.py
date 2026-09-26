#!/usr/bin/env python3
"""Launch a RunPod pod that labels the remaining clips, then terminates itself.

The pod bootstraps from the public GitHub repo (no scp), installs ffmpeg + huggingface_hub into the vLLM
image, runs scripts/label_pod.py under nohup, and shuts itself down when the job prints LABEL_RUN_DONE.
The separate runpod_guard.py on the droplet is the backstop for anything that hangs.

  python3 scripts/launch_label_pod.py --repo-key mommy --budget-h 8 --gpu "A100 PCIe"
"""
from __future__ import annotations
import argparse, json, os, time, urllib.request

API = "https://api.runpod.io/graphql"
GIT = "https://github.com/aloshdenny/text2asmr"


def gql(query: str, key: str) -> dict:
    req = urllib.request.Request(f"{API}?api_key={key}", data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "t2a-launch/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read())
    if "errors" in d: raise RuntimeError(str(d["errors"])[:400])
    return d["data"]


JOBS = {
    # name -> (command tail, whether vLLM is needed). The YouTube-window job only decodes audio and writes a
    # manifest, so it skips the 60 GB model download and runs on the cheapest GPU on offer.
    "label": ("python /workspace/t2a/scripts/label_pod.py --repo-key {repo_key} --budget-h {budget_h} "
              "--shard {shard} --n-shards {n_shards} --concurrency {concurrency}", True),
    "ytwin": ("python /workspace/t2a/scripts/prep_yt_chapters.py --out /workspace/ytwin --manifest-only "
              "--workers 12 --per-chapter 60 --per-class 40000 --push-to aoxo/clap-ft-data "
              "--push-path yt_windows/windows.jsonl --tmp /workspace/tmp_yt", False),
}


def bootstrap(repo_key: str, budget_h: float, shard: int, n_shards: int, concurrency: int, job: str = "label") -> str:
    """One shell command; it must be idempotent because RunPod re-runs it if the container restarts."""
    tail, need_vllm = JOBS[job]
    steps = [
        "set -x",
        "(bash /start.sh > /workspace/start.log 2>&1 &) || service ssh start || true",   # keep sshd: logs matter
        "nvidia-smi",
        "apt-get update -qq && apt-get install -y -qq ffmpeg git ninja-build",
        "pip install -q --no-input uv numpy 'huggingface_hub>=0.25' soundfile",
    ]
    if need_vllm:
        steps.append("uv pip install --system --index-strategy unsafe-best-match "
                     "'https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl' "
                     "--extra-index-url https://download.pytorch.org/whl/cu129 qwen-omni-utils soundfile 'huggingface_hub>=0.25' ninja")
    steps += [
        f"(test -d /workspace/t2a || git clone --depth 1 {GIT} /workspace/t2a)",
        "cd /workspace/t2a && git pull -q || true",
        "export PYTHONPATH=/workspace/t2a HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 T2A_DIR=/workspace/t2a VLLM_USE_FLASHINFER_SAMPLER=0",
        "(" + tail.format(repo_key=repo_key, budget_h=budget_h, shard=shard, n_shards=n_shards,
                          concurrency=concurrency) + " 2>&1 | tee /workspace/job.log; "
        "runpodctl stop pod $RUNPOD_POD_ID || true)",
    ]
    return " && ".join(steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-key", default="mommy")
    ap.add_argument("--budget-h", type=float, default=8.0)
    ap.add_argument("--gpu", default="A100 PCIe", help="gpuType displayName, e.g. 'A100 PCIe' ($1.19/h) or 'H100 PCIe'")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--disk", type=int, default=120, help="container disk GB; clips are cut and deleted per chunk")
    ap.add_argument("--image", default="runpod/pytorch:1.4.0-rc.164-cu1290-torch291-ubuntu2204")
    ap.add_argument("--max-price", type=float, default=2.0, help="refuse to launch above this $/h")
    ap.add_argument("--job", default="label", choices=sorted(JOBS))
    ap.add_argument("--name", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    key = os.environ["RUNPOD_API_KEY"]; hf = os.environ["HF_TOKEN"]

    gpus = gql("""{ gpuTypes { id displayName memoryInGb
        lowestPrice(input:{gpuCount:1}) { uninterruptablePrice } } }""", key)["gpuTypes"]
    match = [g for g in gpus if g["displayName"] == a.gpu] or [g for g in gpus if a.gpu.lower() in g["displayName"].lower()]
    if not match: raise SystemExit(f"no gpuType matching {a.gpu!r}")
    g = match[0]; price = g["lowestPrice"]["uninterruptablePrice"]
    if price is None or price > a.max_price:
        raise SystemExit(f"{g['displayName']} is ${price}/h, above --max-price ${a.max_price}")
    name = a.name or (f"t2a-label-{a.repo_key}-{a.shard}" if a.job == "label" else f"t2a-{a.job}-0")
    print(f"{g['displayName']} {g['memoryInGb']} GB at ${price}/h -> pod {name}, budget {a.budget_h} h "
          f"(max ~${price * a.budget_h:.2f})")

    env = [{"key": "HF_TOKEN", "value": hf}, {"key": "HF_HUB_DISABLE_XET", "value": "1"}]
    pk = os.environ.get("PUBLIC_KEY", "")
    if not pk:
        for cand in (os.path.expanduser("~/.ssh/id_ed25519.pub"), os.path.expanduser("~/.ssh/id_rsa.pub")):
            if os.path.exists(cand):
                pk = open(cand).read().strip(); break      # so the pod's own log can be read, not just its GPU meter
    if pk: env.append({"key": "PUBLIC_KEY", "value": pk})
    envs = ", ".join("{key: \"%s\", value: \"%s\"}" % (e["key"], e["value"]) for e in env)
    cmd = bootstrap(a.repo_key, a.budget_h, a.shard, a.n_shards, a.concurrency, a.job).replace('"', '\\"')

    mutation = f'''mutation {{ podFindAndDeployOnDemand(input: {{
        cloudType: ALL, gpuCount: 1, gpuTypeId: "{g['id']}", name: "{name}",
        imageName: "{a.image}", containerDiskInGb: {a.disk}, volumeInGb: 0, minMemoryInGb: 60, minVcpuCount: 12,
        dockerArgs: "bash -lc \\"{cmd}\\"", ports: "8000/http,22/tcp",
        env: [{envs}] }}) {{ id name machineId costPerHr }} }}'''
    if a.dry_run:
        print(mutation[:1200]); return 0
    pod = gql(mutation, key)["podFindAndDeployOnDemand"]
    print(f"launched {pod['name']} id={pod['id']} ${pod['costPerHr']}/h")
    print("watch:  ssh root@139.59.33.163 'tail -f /root/t2a/runpod_guard.log'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
