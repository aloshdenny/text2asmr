#!/bin/bash
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^HF_TOKEN=" | xargs); export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
cd /workspace/t2a; mkdir -p /workspace/lab
python -m pip install -q -U "transformers<5" soundfile "numpy<2" 2>&1 | grep -vE "WARNING|notice" | tail -1
echo "== candidates + prep (CLAP gate)"; python scripts/label_audios2_qwen3.py --stage candidates,prep --workers 20 --bg-max 0.5 > /workspace/lab/prep.log 2>&1 || true
grep -E "candidates:|PREP_DONE|Traceback" /workspace/lab/prep.log | tail -3 | cut -c1-300
echo "== vLLM env"; (python -m venv /workspace/vllm && /workspace/vllm/bin/pip install -q -U pip && /workspace/vllm/bin/pip install -q vllm qwen-omni-utils soundfile) > /workspace/lab/vllm_install.log 2>&1 || true
(/workspace/vllm/bin/vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --dtype bfloat16 --max-model-len 4096 --limit-mm-per-prompt '{"audio":1}' --gpu-memory-utilization 0.92 --port 8000 > /workspace/lab/vllm.log 2>&1 &)
for i in $(seq 1 60); do curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 20; done
curl -s http://127.0.0.1:8000/v1/models | head -c 200; echo
echo "== label (vLLM if up, else HF batched)"
# periodic uploader (every 30 min) alongside labeling
(while true; do sleep 1800; python scripts/label_audios2_qwen3.py --stage upload >> /workspace/lab/upload.log 2>&1; done) &
if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then python scripts/label_audios2_qwen3.py --stage label > /workspace/lab/label.log 2>&1
else /workspace/venv5/bin/python scripts/label_audios2_qwen3.py --stage label > /workspace/lab/label.log 2>&1; fi
python scripts/label_audios2_qwen3.py --stage upload >> /workspace/lab/upload.log 2>&1
echo PIPELINE_DONE
