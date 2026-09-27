#!/bin/bash
# CLAP v7 stage 2 on a RunPod pod: everything from the Hub, nothing from tinkerspace.
#   v7_mels/*             in-domain vocal mels  -> shards 0..999
#   yt_mels/ytmels.tar    YouTube chapter mels  -> shards +1000
#   v7s2/index.jsonl      merged, precision-weighted index (build_v7_stage2.py)
#   v7_ckpt/stage1_best   warm start
# Pushes the best checkpoint and the training log back to aoxo/clap-ft-data/v7_ckpt/stage2_best.
set -euo pipefail
W=/workspace; R=aoxo/clap-ft-data
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q "transformers==4.46.3"
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
for i in range(6):
    try:
        snapshot_download("aoxo/clap-ft-data", repo_type="dataset", local_dir="/workspace/hub",
                          allow_patterns=["v7_mels/*", "v7s2/index.jsonl", "v7_ckpt/stage1_best/*", "yt_mels/ytmels.tar"],
                          max_workers=8)
        break
    except Exception as e:
        print("snapshot retry", i, type(e).__name__, str(e)[:120], flush=True)
print("DOWNLOADED", flush=True)
PY
mkdir -p $W/yt && tar -xf $W/hub/yt_mels/ytmels.tar -C $W/yt && rm -f $W/hub/yt_mels/ytmels.tar
YT=$(dirname "$(find $W/yt -name index.jsonl | head -1)")
mkdir -p $W/data
for f in $W/hub/v7_mels/shard_*.f16; do ln -sfn "$f" $W/data/$(basename $f); done
for f in $YT/shard_*.f16; do n=$(basename $f .f16); n=$((10#${n#shard_} + 1000)); ln -sfn "$f" $W/data/$(printf "shard_%04d.f16" $n); done
cp $W/hub/v7s2/index.jsonl $W/data/index.jsonl
echo "shards: $(ls $W/data/shard_*.f16 | wc -l)  rows: $(wc -l < $W/data/index.jsonl)"
# every shard the index names must exist, or the first batch that touches it dies an hour in
python - <<'PY'
import json, os
need = {json.loads(l)["shard"] for l in open("/workspace/data/index.jsonl")}
miss = [s for s in need if not os.path.exists(f"/workspace/data/shard_{s:04d}.f16")]
assert not miss, f"missing shards: {sorted(miss)[:10]}"
print(f"all {len(need)} shards present")
PY
python $W/t2a/scripts/train_clap_v3.py --data $W/data --out $W/ckpt --init $W/hub/v7_ckpt/stage1_best \
  --batch 112 --bg-frac 0.3 --max-steps 8000 --lr 2e-5 --text-lr 1e-5 \
  --warmup 100 --eval-every 500 --eval-chunk 64 --augment --precision bf16 2>&1 | tee $W/v7s2.log
grep -q TRAIN_DONE $W/v7s2.log
cp $W/v7s2.log $W/ckpt/best/train.log
python - <<'PY'
from huggingface_hub import HfApi
for i in range(5):
    try:
        HfApi().upload_folder(repo_id="aoxo/clap-ft-data", repo_type="dataset", folder_path="/workspace/ckpt/best",
                              path_in_repo="v7_ckpt/stage2_best", commit_message="CLAP v7 stage 2 best (RunPod)")
        print("PUSHED", flush=True); break
    except Exception as e:
        print("push retry", i, e, flush=True)
else:
    raise SystemExit("push failed")
PY
