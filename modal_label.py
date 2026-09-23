"""Modal app for the remaining Qwen3-Omni labeling, sized to a fixed credit budget.

Runs in whichever workspace has credit (`modal profile activate <ws>; modal deploy modal_label.py`).
Reads the pre-staged candidate lists (`labels/pending_candidates.jsonl` in each corpus repo), labels them in
chunks, and appends each chunk to the corpus ledger on the Hub, so a container that dies (24 h cap, spend
limit, preemption) never loses more than one chunk and the next run resumes from the ledger.

  modal run -d modal_label.py::label_pending --repo-key mommy --budget-h 7
"""
import modal, os, subprocess, time, json
from pathlib import Path

app = modal.App("t2a-label")
HF = modal.Secret.from_name("t2a-hf")
CACHE = modal.Volume.from_name("t2a-cache", create_if_missing=True)
REPO_DIR = Path(__file__).parent
ENV = {"HF_HUB_DISABLE_XET": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root/t2a", "HF_HOME": "/cache/hf"}

label_image = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "ninja-build", "curl")
    .pip_install("uv")
    .run_commands('uv pip install --system --index-strategy unsafe-best-match '
                  '"https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" '
                  '--extra-index-url https://download.pytorch.org/whl/cu129 qwen-omni-utils soundfile huggingface_hub ninja')
    .env({**ENV, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(str(REPO_DIR / "scripts"), remote_path="/root/t2a/scripts")
    .add_local_dir(str(REPO_DIR / "text2asmr"), remote_path="/root/t2a/text2asmr"))

CORPUS = {"mommy": ("aoxo/t2a-mommy", "labels/qwen3omni_expansion.jsonl"),
          "daddy": ("aoxo/t2a-daddy", "labels/qwen3omni.jsonl")}
PENDING = "labels/pending_candidates.jsonl"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _serve():
    """Start vLLM and wait for it; returns the process."""
    import urllib.request
    srv = subprocess.Popen(["vllm", "serve", MODEL, "--dtype", "bfloat16", "--max-model-len", "4096",
                            "--limit-mm-per-prompt", '{"audio":1}', "--gpu-memory-utilization", "0.90", "--port", "8000"],
                           stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(160):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5); print("vllm up", flush=True); return srv
        except Exception:
            if srv.poll() is not None:
                print(open("/tmp/vllm.log").read()[-4000:], flush=True); raise SystemExit("vllm died during startup")
            time.sleep(15)
    raise SystemExit("vllm did not come up in 40 min")


@app.function(image=label_image, gpu="A100-80GB", cpu=12, memory=65536, timeout=23 * 3600,
              secrets=[HF], volumes={"/cache": CACHE})
def label_pending(repo_key: str = "mommy", budget_h: float = 7.0, chunk: int = 40000, concurrency: int = 64):
    """Label the pre-staged candidates for one corpus until they run out or the time budget ends."""
    from huggingface_hub import HfApi, hf_hub_download
    repo, ledger = CORPUS[repo_key]
    api = HfApi(); work = Path("/tmp/lab"); work.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); budget_s = budget_h * 3600

    done = set()
    for r, l in (CORPUS["mommy"], CORPUS["daddy"], (CORPUS["mommy"][0], "labels/qwen3omni.jsonl")):
        try:
            for line in open(hf_hub_download(r, l, repo_type="dataset", force_download=True)):
                done.add(json.loads(line)["uid"])
        except Exception: pass
    print(f"{len(done)} uids already labeled", flush=True)

    pend = [json.loads(l) for l in open(hf_hub_download(repo, PENDING, repo_type="dataset", force_download=True))]
    todo = [r for r in pend if r["uid"] not in done]
    print(f"{repo}: {len(pend)} pre-staged, {len(todo)} still to label", flush=True)
    if not todo: return 0

    # chunk along source boundaries: stage_prep downloads per source, so splitting a source across
    # chunks would fetch it twice
    from collections import defaultdict
    by_src = defaultdict(list)
    for r in todo: by_src[r["source"]].append(r)
    chunks, cur = [], []
    for src in by_src:
        cur.extend(by_src[src])
        if len(cur) >= chunk: chunks.append(cur); cur = []
    if cur: chunks.append(cur)
    print(f"{len(chunks)} chunks over {len(by_src)} sources", flush=True)

    srv = _serve(); labeled_total = 0
    for ci, part in enumerate(chunks):
        left = budget_s - (time.time() - t0)
        if left < 1800:
            print(f"budget spent ({(time.time()-t0)/3600:.1f} h); stopping cleanly", flush=True); break
        (work / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
        # per-chunk state: prep resumes per source, so its done-list must not leak across chunks
        for f in ("labels.jsonl", "clap_index.jsonl", "prep_done_sources.txt"): (work / f).unlink(missing_ok=True)
        print(f"chunk {ci}/{len(chunks)}: {len(part)} clips, {left/3600:.1f} h of budget left", flush=True)
        subprocess.run(["python", "/root/t2a/scripts/label_audios2_qwen3.py", "--stage", "prep", "--work", str(work),
                        "--workers", "12", "--bg-max", "1.01", "--no-clap"], cwd="/root/t2a", check=True)
        subprocess.run(["python", "/root/t2a/scripts/label_audios2_qwen3.py", "--stage", "label", "--work", str(work),
                        "--concurrency", str(concurrency), "--delete-wav", "--no-clap"], cwd="/root/t2a", check=True)

        new = [json.loads(l) for l in open(work / "labels.jsonl")] if (work / "labels.jsonl").exists() else []
        idx = {r["uid"]: r for r in part}
        rows = [dict(r, source=idx[r["uid"]]["source"], repo=repo) for r in new if r["uid"] in idx]
        if not rows: print("chunk produced no rows", flush=True); continue
        old = []
        try: old = [json.loads(l) for l in open(hf_hub_download(repo, ledger, repo_type="dataset", force_download=True))]
        except Exception: pass
        out = work / "ledger.jsonl"; out.write_text("".join(json.dumps(r) + "\n" for r in old + rows))
        for attempt in range(5):
            try:
                api.upload_file(path_or_fileobj=str(out), path_in_repo=ledger, repo_id=repo, repo_type="dataset",
                                commit_message=f"+{len(rows)} Qwen3-Omni labels"); break
            except Exception as e:
                print(f"upload retry {attempt}: {type(e).__name__} {str(e)[:120]}", flush=True); time.sleep(30 * (attempt + 1))
        labeled_total += len(rows)
        print(f"uploaded {ledger}: +{len(rows)} (run total {labeled_total}, ledger {len(old) + len(rows)})", flush=True)
    srv.kill(); print(f"LABEL_RUN_DONE labeled={labeled_total} in {(time.time()-t0)/3600:.2f} h", flush=True)
    return labeled_total
