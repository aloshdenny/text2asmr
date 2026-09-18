#!/bin/bash
set -e
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^HF_TOKEN=" | xargs); export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/t2a; mkdir -p /workspace/v5
[ -f /workspace/v5/labeled.jsonl ] || python -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('aoxo/clap-ft-data','v2/selftrain_labeled.jsonl',repo_type='dataset'),'/workspace/v5/labeled.jsonl')"
echo "== A: Pro vocal mel"; grep -q PREP_DONE /workspace/v5/prepA.log 2>/dev/null || python scripts/prep_clap_v2.py --subset /workspace/v5/labeled.jsonl --out /workspace/v5/mel_pro --workers 24 > /workspace/v5/prepA.log 2>&1
echo "== B: YouTube chapter windows (all videos)"; grep -q PREP_DONE /workspace/v5/prepB.log 2>/dev/null || python scripts/prep_yt_chapters.py --out /workspace/v5/mel_yt --workers 6 --per-chapter 60 --per-class 20000 --bg-per-video 60 > /workspace/v5/prepB.log 2>&1
grep -E "videos with|PREP_DONE" /workspace/v5/prepB.log | cut -c1-300
echo "== C: merge"; mkdir -p /workspace/v5/mel_all; rm -f /workspace/v5/mel_all/*
python - <<'PY'
import json, glob, os
n = 0; rows = []
for d in ("/workspace/v5/mel_pro", "/workspace/v5/mel_yt"):
    shards = sorted(glob.glob(f"{d}/shard_*.f16")); off = n
    for p in shards: os.symlink(p, f"/workspace/v5/mel_all/shard_{n:04d}.f16"); n += 1
    for l in open(f"{d}/index.jsonl"):
        r = json.loads(l); r["shard"] += off; rows.append(r)
with open("/workspace/v5/mel_all/index.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("merged rows:", len(rows))
PY
echo "== D: train v5 (MIL)"; python scripts/train_clap_v5.py --data /workspace/v5/mel_all --out /workspace/v5/model --bags 12 --k 6 --topk 3 --max-steps 3000 --eval-every 200 --augment > /workspace/v5/train.log 2>&1
grep -E "EVAL" /workspace/v5/train.log | tail -3 | cut -c1-300
echo "== E: push"; python - <<'PY'
import json; from huggingface_hub import HfApi
evs = [json.loads(l) for l in open("/workspace/v5/model/eval.jsonl")]; b = max(evs, key=lambda e: e["yt_chapter_acc"] * 0.5 + e["yt_macro"] * 0.5)
HfApi().upload_folder(folder_path="/workspace/v5/model/best", repo_id="aoxo/clap-htsat-unfused-asmr-v2", repo_type="model", path_in_repo="v5", commit_message=f"CLAP v5 MIL: yt chapter acc={b['yt_chapter_acc']:.3f} macro={b['yt_macro']:.3f} pro={b['pro_window_acc']}")
HfApi().upload_file(path_or_fileobj="/workspace/v5/model/eval.jsonl", path_in_repo="v5/eval.jsonl", repo_id="aoxo/clap-htsat-unfused-asmr-v2", repo_type="model"); print("pushed")
PY
echo PIPELINE_DONE
