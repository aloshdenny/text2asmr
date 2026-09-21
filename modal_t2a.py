"""Modal deployment for text2asmr background jobs. Every worker is stateless, resumes from the Hub, runs <= 23 h, and is
rescheduled daily (Modal containers cap at 24 h). Deploy: `modal deploy modal_t2a.py`; kick a worker now: `modal run modal_t2a.py::<fn>`."""
import modal, os, subprocess, time, json
from pathlib import Path

app = modal.App("t2a")
HF = modal.Secret.from_name("t2a-hf")
CACHE = modal.Volume.from_name("t2a-cache", create_if_missing=True)
REPO_DIR = Path(__file__).parent
ENV = {"HF_HUB_DISABLE_XET": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root/t2a", "HF_HOME": "/cache/hf", "T2A_WHISPER_BATCH": "16", "T2A_TW": "2"}

transcribe_image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg")
    .pip_install("faster-whisper", "huggingface_hub>=0.25", "numpy<2", "nvidia-cudnn-cu12", "nvidia-cublas-cu12")
    .env(ENV)
    .add_local_dir(str(REPO_DIR / "scripts"), remote_path="/root/t2a/scripts")
    .add_local_dir(str(REPO_DIR / "text2asmr"), remote_path="/root/t2a/text2asmr"))

label_image = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11").apt_install("ffmpeg", "ninja-build", "curl")
    .pip_install("uv")
    .run_commands('uv pip install --system --index-strategy unsafe-best-match "https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" --extra-index-url https://download.pytorch.org/whl/cu129 qwen-omni-utils soundfile huggingface_hub ninja')
    .env({**ENV, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(str(REPO_DIR / "scripts"), remote_path="/root/t2a/scripts")
    .add_local_dir(str(REPO_DIR / "text2asmr"), remote_path="/root/t2a/text2asmr"))

def _ld():
    import nvidia.cublas.lib, nvidia.cudnn.lib
    return list(nvidia.cublas.lib.__path__)[0] + ":" + list(nvidia.cudnn.lib.__path__)[0]

def _transcribe(repo, extra, budget_s=23 * 3600 - 900):
    env = {**os.environ, "LD_LIBRARY_PATH": _ld() + ":" + os.environ.get("LD_LIBRARY_PATH", "")}; t0 = time.time()
    while time.time() - t0 < budget_s:
        rc = subprocess.run(["python", "/root/t2a/scripts/transcribe_audios2.py", "--repo", repo, "--model", "large-v3", "--compute-type", "float16", "--transcribe-workers", os.environ.get("T2A_TW", "2"), "--producer-workers", "4",
                             "--upload-batch-size", "64", "--upload-batch-timeout", "1500", *extra], cwd="/root/t2a", env=env).returncode
        print(f"transcribe {repo} {extra} exited rc={rc}; ", "done" if rc == 0 else "retrying in 60s", flush=True)
        if rc == 0: return
        time.sleep(60)

TR = dict(image=transcribe_image, gpu="L4", cpu=8, memory=32768, timeout=23 * 3600, secrets=[HF], volumes={"/cache": CACHE})
@app.function(**TR)
def audios3_shard0(): _transcribe("aoxo/audios3", ["--num-shards", "4", "--shard-index", "0"])
@app.function(**TR)
def audios3_shard1(): _transcribe("aoxo/audios3", ["--num-shards", "4", "--shard-index", "1"])
@app.function(**TR)
def audios3_shard2(): _transcribe("aoxo/audios3", ["--num-shards", "4", "--shard-index", "2"])
@app.function(**TR)
def audios3_shard3(): _transcribe("aoxo/audios3", ["--num-shards", "4", "--shard-index", "3"])

def _expansion(idx, n):
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    while time.time() - t0 < 23 * 3600 - 1800:
        rows = [json.loads(l) for l in open(hf_hub_download("aoxo/clap-ft-data", "v2/acquired_snapshot.jsonl", repo_type="dataset", force_download=True))]
        for repo in ("audios2", "audios3"):
            cs = [r["uploader"] for r in rows if r["repo"].endswith(repo) and r["files"] > 0]
            Path(f"/tmp/creators_{repo}.txt").write_text("\n".join(cs) + "\n"); print(f"{repo}: {len(cs)} expansion creators (shard {idx}/{n})", flush=True)
            if cs: _transcribe(f"aoxo/{repo}", ["--creators-file", f"/tmp/creators_{repo}.txt", "--num-shards", str(n), "--shard-index", str(idx)], budget_s=6 * 3600)
        print("expansion pass done; sleeping 10 min", flush=True); time.sleep(600)
@app.function(**TR)
def expansion_transcribe(): _expansion(0, 2)
@app.function(**TR)
def expansion_transcribe1(): _expansion(1, 2)

def _old_expansion_transcribe():
    """Continuously transcribe newly-acquired creators (both repos), refreshing the creator list from the acquisition ledger."""
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    while time.time() - t0 < 23 * 3600 - 1800:
        rows = [json.loads(l) for l in open(hf_hub_download("aoxo/clap-ft-data", "v2/acquired_snapshot.jsonl", repo_type="dataset", force_download=True))]
        for repo in ("audios2", "audios3"):
            cs = [r["uploader"] for r in rows if r["repo"].endswith(repo) and r["files"] > 0]
            Path(f"/tmp/creators_{repo}.txt").write_text("\n".join(cs) + "\n"); print(f"{repo}: {len(cs)} expansion creators", flush=True)
            if cs: _transcribe(f"aoxo/{repo}", ["--creators-file", f"/tmp/creators_{repo}.txt"], budget_s=6 * 3600)
        print("expansion pass done; sleeping 10 min", flush=True); time.sleep(600)

@app.function(image=label_image, gpu="A100-80GB", cpu=12, memory=65536, timeout=8 * 3600, secrets=[HF], volumes={"/cache": CACHE})
def label_new_transcripts():
    """Daily: gap clips from transcripts not yet labeled (expansion creators in both repos + audios3) -> Qwen3-Omni via vLLM -> HF ledgers."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); work = Path("/tmp/lab"); work.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen(["vllm", "serve", "Qwen/Qwen3-Omni-30B-A3B-Instruct", "--dtype", "bfloat16", "--max-model-len", "4096", "--limit-mm-per-prompt", '{"audio":1}', "--gpu-memory-utilization", "0.85", "--port", "8000"], stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)
    labeled = set()
    for ledger in ("v2/audios2_qwen3omni_labels.jsonl", "v2/audios3_qwen3omni_labels.jsonl", "v2/expansion_qwen3omni_labels.jsonl"):
        try:
            for l in open(hf_hub_download("aoxo/clap-ft-data", ledger, repo_type="dataset", force_download=True)): labeled.add(json.loads(l)["uid"])
        except Exception: pass
    print("already labeled uids:", len(labeled), flush=True)
    # candidate files: any transcribed (.json) source in audios3, or in audios2 belonging to an expansion creator
    acq = [json.loads(l) for l in open(hf_hub_download("aoxo/clap-ft-data", "v2/acquired_snapshot.jsonl", repo_type="dataset", force_download=True))]
    exp2 = {r["uploader"] for r in acq if r["repo"].endswith("audios2")}
    files = {}
    for repo in ("aoxo/audios2", "aoxo/audios3"):
        fs = api.list_repo_files(repo, repo_type="dataset"); js = {f[:-5] for f in fs if f.endswith(".json")}
        sel = [f for f in fs if f.endswith(".m4a") and f in js and (repo.endswith("audios3") or f.split("/")[0] in exp2)]
        files[repo] = sel; print(repo, "transcribed candidate files:", len(sel), flush=True)
    cand = work / "candidates.jsonl"; cand.unlink(missing_ok=True)
    for repo, sel in files.items():
        lst = work / f"files_{repo.split('/')[-1]}.txt"; lst.write_text("\n".join(sel) + "\n")
        subprocess.run(["python", "/root/t2a/scripts/candidates_from_transcripts.py", "--repo", repo, "--files", str(lst), "--out", str(cand)], cwd="/root/t2a", check=True)
    rows = [json.loads(l) for l in open(cand)] if cand.exists() else []; rows = [r for r in rows if r["uid"] not in labeled]
    cand.write_text("".join(json.dumps(r) + "\n" for r in rows)); print("new candidate clips:", len(rows), flush=True)
    if not rows: srv.kill(); return
    subprocess.run(["python", "/root/t2a/scripts/label_audios2_qwen3.py", "--stage", "prep", "--work", str(work), "--workers", "12", "--bg-max", "1.01", "--no-clap"], cwd="/root/t2a", check=True)
    import urllib.request
    for _ in range(120):
        try: urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5); break
        except Exception: time.sleep(15)
    subprocess.run(["python", "/root/t2a/scripts/label_audios2_qwen3.py", "--stage", "label", "--work", str(work), "--concurrency", "48", "--delete-wav", "--no-clap"], cwd="/root/t2a", check=True)
    # merge into per-repo ledgers on the Hub (append semantics via download+concat)
    new = [json.loads(l) for l in open(work / "labels.jsonl")]; idx = {json.loads(l)["uid"]: json.loads(l) for l in open(work / "clap_index.jsonl")}
    for repo_key, ledger in (("audios3", "v2/audios3_qwen3omni_labels.jsonl"), ("audios2", "v2/expansion_qwen3omni_labels.jsonl")):
        part = [dict(r, source=idx[r["uid"]]["source"], repo=idx[r["uid"]]["repo"]) for r in new if r["uid"] in idx and idx[r["uid"]]["repo"].endswith(repo_key)]
        if not part: continue
        old = []
        try: old = [json.loads(l) for l in open(hf_hub_download("aoxo/clap-ft-data", ledger, repo_type="dataset", force_download=True))]
        except Exception: pass
        out = work / ledger.split("/")[-1]; out.write_text("".join(json.dumps(r) + "\n" for r in old + part))
        api.upload_file(path_or_fileobj=str(out), path_in_repo=ledger, repo_id="aoxo/clap-ft-data", repo_type="dataset", commit_message=f"+{len(part)} Qwen3-Omni labels ({repo_key})")
        print(f"uploaded {ledger}: +{len(part)} (total {len(old) + len(part)})", flush=True)
    srv.kill(); print("LABEL_BATCH_DONE", flush=True)

WORKERS = ["audios3_shard0", "audios3_shard1", "audios3_shard2", "audios3_shard3", "expansion_transcribe", "expansion_transcribe1", "label_new_transcripts"]

@app.function(image=modal.Image.debian_slim(python_version="3.11"), schedule=modal.Period(hours=24), timeout=600)
def dispatcher():
    """The only scheduled function (plan limit): re-spawns every worker daily; workers self-limit to 23 h and resume from the Hub."""
    for name in WORKERS:
        fc = modal.Function.from_name("t2a", name).spawn(); print("spawned", name, fc.object_id, flush=True)

@app.local_entrypoint()
def main():
    print("deploy with: modal deploy modal_t2a.py ; kick now with: modal run --detach modal_t2a.py::audios3_shard0 (etc.)")
