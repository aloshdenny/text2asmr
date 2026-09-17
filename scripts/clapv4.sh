#!/bin/bash
set -e
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^HF_TOKEN=" | xargs); export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/t2a; mkdir -p /workspace/v4
[ -f /workspace/v4/labeled.jsonl ] || python -c "from huggingface_hub import hf_hub_download as d; import shutil, os; shutil.copy(d('aoxo/clap-ft-data','v2/selftrain_labeled.jsonl',repo_type='dataset'),'/workspace/v4/labeled.jsonl')"
echo "== A: mel for Pro-labeled vocal set"; grep -q PREP_DONE /workspace/v4/prepA.log 2>/dev/null || python scripts/prep_clap_v2.py --subset /workspace/v4/labeled.jsonl --out /workspace/v4/mel_pro --workers 24 > /workspace/v4/prepA.log 2>&1
echo "== B: YouTube chapter windows"; grep -q PREP_DONE /workspace/v4/prepB.log 2>/dev/null || python scripts/prep_yt_chapters.py --out /workspace/v4/mel_yt --workers 6 > /workspace/v4/prepB.log 2>&1
grep PREP_DONE /workspace/v4/prepB.log | cut -c1-400
echo "== C: merge"; mkdir -p /workspace/v4/mel_all; rm -f /workspace/v4/mel_all/*
python - <<'PY'
import json, glob, os
n = 0; rows = []
for d in ("/workspace/v4/mel_pro", "/workspace/v4/mel_yt"):
    shards = sorted(glob.glob(f"{d}/shard_*.f16")); off = n
    for p in shards: os.symlink(p, f"/workspace/v4/mel_all/shard_{n:04d}.f16"); n += 1
    for l in open(f"{d}/index.jsonl"):
        r = json.loads(l); r["shard"] += off; rows.append(r)
with open("/workspace/v4/mel_all/index.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
from collections import Counter; print("merged rows:", len(rows), Counter((r["split"], r["label"]) for r in rows).most_common(40))
PY
echo "== D: train v4"; python scripts/train_clap_v3.py --data /workspace/v4/mel_all --out /workspace/v4/model --batch 64 --bg-frac 0.4 --max-steps 4000 --eval-every 200 --augment --ce-weight 1.0 > /workspace/v4/train.log 2>&1
grep -E "HEAD|EVAL" /workspace/v4/train.log | tail -2 | cut -c1-400
echo "== E: push"; python - <<'PY'
import json; from huggingface_hub import HfApi
evs = [json.loads(l) for l in open("/workspace/v4/model/eval.jsonl")]; b = max(evs, key=lambda e: (e["head"]["p85"]["recall"] if e["head"]["p85"] else 0) + 0.1 * e["head"]["acc4"])
HfApi().upload_folder(folder_path="/workspace/v4/model/best", repo_id="aoxo/clap-htsat-unfused-asmr-v2", repo_type="model", path_in_repo="v4", commit_message=f"CLAP v4: Pro vocal + YouTube chapter labels; acc4={b['head']['acc4']:.3f} bg_recall={b['head']['bg_recall']:.3f} acc={b['acc3']:.3f}")
HfApi().upload_file(path_or_fileobj="/workspace/v4/model/eval.jsonl", path_in_repo="v4/eval.jsonl", repo_id="aoxo/clap-htsat-unfused-asmr-v2", repo_type="model")
HfApi().upload_file(path_or_fileobj="/workspace/v4/mel_yt/index.jsonl", path_in_repo="v2/yt_chapter_windows_index.jsonl", repo_id="aoxo/clap-ft-data", repo_type="dataset", commit_message="YouTube chapter windows (labels, uids, rms)")
print("pushed")
PY
echo PIPELINE_DONE
