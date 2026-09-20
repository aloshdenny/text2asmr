#!/bin/bash
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^HF_TOKEN=" | xargs); export HF_HUB_DISABLE_XET=1 VLLM_USE_FLASHINFER_SAMPLER=0
apt-get install -y -qq ninja-build >/dev/null 2>&1 || true
cd /workspace && python -m venv /workspace/vllm && /workspace/vllm/bin/pip install -q -U pip uv ninja
V=0.29.0; /workspace/vllm/bin/uv pip install --python /workspace/vllm/bin/python "https://github.com/vllm-project/vllm/releases/download/v${V}/vllm-${V}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match qwen-omni-utils soundfile
export PATH=/workspace/vllm/bin:$PATH
exec /workspace/vllm/bin/vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --dtype bfloat16 --max-model-len 4096 --limit-mm-per-prompt '{"audio":1}' --gpu-memory-utilization 0.85 --port 8000
