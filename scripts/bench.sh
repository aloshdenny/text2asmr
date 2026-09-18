#!/bin/bash
set -e
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^HF_TOKEN=" | xargs); export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/t2a
python -m pip install -q -U "transformers<5" accelerate soundfile librosa "numpy<2" torchvision 2>&1 | grep -vE "WARNING|notice" | tail -1
python -c "import transformers; print('transformers', transformers.__version__)"
python scripts/bench_audio_llms.py --stage cut,label,score > /workspace/bench.log 2>&1 || true
tail -5 /workspace/bench.log | cut -c1-300
python - <<'PY'
from huggingface_hub import HfApi, CommitOperationAdd; import glob, os
ops = [CommitOperationAdd(path_in_repo=f"pilot/bench/{os.path.basename(p)}", path_or_fileobj=p) for p in glob.glob("/workspace/bench/labels_*.jsonl") + glob.glob("/workspace/bench/report.json") + glob.glob("/workspace/bench/clips.json")]
if ops: HfApi().create_commit(repo_id="aoxo/clap-ft-data", repo_type="dataset", operations=ops, commit_message="open audio-LLM labeler benchmark vs Gemini-Pro (pilot clips)"); print("uploaded", len(ops))
PY
echo PIPELINE_DONE
