#!/bin/bash
set -e
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^HF_TOKEN=" | xargs); export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
cd /workspace && python -m venv venv5 && ./venv5/bin/pip install -q -U pip && ./venv5/bin/pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && ./venv5/bin/pip install -q -U transformers accelerate soundfile librosa "numpy<2" huggingface_hub qwen-omni-utils 2>&1 | grep -vE "WARNING|notice" | tail -1
./venv5/bin/python -c "import transformers; print('transformers', transformers.__version__)"
mkdir -p /workspace/bench && cd /workspace/bench && ./../venv5/bin/python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('aoxo/clap-ft-data','pilot/bench/clips_wav.tar',repo_type='dataset'),'clips_wav.tar')" && tar -xf clips_wav.tar && sed -i 's#"/workspace/bench/wav/#"/workspace/bench/wav/#' clips.json && ls wav | wc -l
cd /workspace/t2a && /workspace/venv5/bin/python scripts/bench_audio_llms.py --stage label --models qwen3-omni-30b,qwen3-omni-captioner > /workspace/bench3.log 2>&1 || true
grep -E "parsed in|FAILED" /workspace/bench3.log | cut -c1-300
/workspace/venv5/bin/python - <<'PY'
from huggingface_hub import HfApi, CommitOperationAdd; import glob, os
ops = [CommitOperationAdd(path_in_repo=f"pilot/bench/{os.path.basename(p)}", path_or_fileobj=p) for p in glob.glob("/workspace/bench/labels_qwen3*.jsonl")]
if ops: HfApi().create_commit(repo_id="aoxo/clap-ft-data", repo_type="dataset", operations=ops, commit_message="Qwen3-Omni labeler bench"); print("uploaded", len(ops))
PY
echo PIPELINE_DONE
