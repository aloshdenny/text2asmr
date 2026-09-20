#!/bin/bash
# Sample newly-acquired creators, transcribe, cut gaps, label with Qwen3-Omni (vLLM), report per-label yield vs audios2 baseline.
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^HF_TOKEN=" | xargs); export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/t2a
cd /workspace/t2a; mkdir -p /workspace/yc; apt-get install -y -qq ffmpeg >/dev/null 2>&1
python -m pip install -q -U faster-whisper "transformers<5" soundfile "numpy<2" nvidia-cudnn-cu12 nvidia-cublas-cu12 2>&1 | grep -vE "WARNING|notice" | tail -1
export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas.lib, nvidia.cudnn.lib; print(list(nvidia.cublas.lib.__path__)[0] + ':' + list(nvidia.cudnn.lib.__path__)[0])"):$LD_LIBRARY_PATH
python - <<'PY'
import json, random, os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); acq = [json.loads(l) for l in open(hf_hub_download("aoxo/clap-ft-data", "v2/acquired_snapshot.jsonl", repo_type="dataset"))]
rng = random.Random(0)
for repo in ("aoxo/audios2", "aoxo/audios3"):
    cs = [r["uploader"] for r in acq if r["repo"] == repo and r["files"] >= 20]; rng.shuffle(cs); cs = cs[:3]
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".m4a") and f.split("/")[0] in cs]
    rng.shuffle(files); files = files[:120]
    open(f"/workspace/yc/creators_{repo.split('/')[-1]}.txt", "w").write("\n".join(cs) + "\n"); open(f"/workspace/yc/files_{repo.split('/')[-1]}.txt", "w").write("\n".join(files) + "\n")
    print(repo, "creators", cs, "files", len(files))
PY
for r in audios2 audios3; do
  python scripts/transcribe_audios2.py --repo aoxo/$r --model large-v3 --compute-type float16 --transcribe-workers 4 --producer-workers 4 --upload-batch-size 32 --upload-batch-timeout 600 --creators-file /workspace/yc/creators_$r.txt > /workspace/yc/transcribe_$r.log 2>&1
  grep -cE "entries \(speech" /workspace/yc/transcribe_$r.log
  python scripts/candidates_from_transcripts.py --repo aoxo/$r --files /workspace/yc/files_$r.txt --out /workspace/yc/candidates.jsonl
done
mkdir -p /workspace/lab && cp /workspace/yc/candidates.jsonl /workspace/lab/candidates.jsonl
python scripts/label_audios2_qwen3.py --stage prep --work /workspace/lab --workers 12 --bg-max 1.01 > /workspace/yc/prep.log 2>&1; grep PREP_DONE /workspace/yc/prep.log | cut -c1-120
bash /workspace/start_vllm_cu129.sh > /workspace/yc/vllm.log 2>&1 &
for i in $(seq 1 90); do curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 20; done
python scripts/label_audios2_qwen3.py --stage label --work /workspace/lab --concurrency 48 > /workspace/yc/label.log 2>&1; grep LABEL_DONE /workspace/yc/label.log
python - <<'PY'
import json, collections
idx = {json.loads(l)["uid"]: json.loads(l) for l in open("/workspace/lab/clap_index.jsonl")}
rows = [json.loads(l) for l in open("/workspace/lab/labels.jsonl")]; by = collections.defaultdict(collections.Counter)
for r in rows:
    if r.get("label"): by[idx[r["uid"]]["repo"]][r["label"]] += 1
base = {"whispering": 65.1, "normal speech": 11.6, "breathing": 10.0, "mouth sounds": 7.1, "moaning": 3.4, "kissing": 1.9, "silence": 0.5}
for repo, c in by.items():
    n = sum(c.values()); print(f"\n{repo}: {n} gap clips from new creators")
    for k, v in c.most_common(10): print(f"  {k:14s} {100*v/n:5.1f}%   (audios2 baseline {base.get(k, 0):4.1f}%)")
PY
python -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/workspace/lab/labels.jsonl', path_in_repo='v2/yield_check_labels.jsonl', repo_id='aoxo/clap-ft-data', repo_type='dataset'); print('uploaded')"
echo PIPELINE_DONE
