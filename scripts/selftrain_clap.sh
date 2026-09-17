#!/bin/bash
# Pod-side pipeline: prep labeled -> teacher -> prep pool -> pseudo-label -> student -> push. Resumable stage by stage.
set -e
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^HF_TOKEN=" | xargs); export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/t2a; mkdir -p /workspace/st
[ -f /workspace/st/labeled.jsonl ] || python -c "from huggingface_hub import hf_hub_download as d; import shutil, os; shutil.copy(d('aoxo/clap-ft-data','v2/selftrain_labeled.jsonl',repo_type='dataset',token=os.environ['HF_TOKEN']),'/workspace/st/labeled.jsonl'); shutil.copy(d('aoxo/clap-ft-data','v2/selftrain_pool.jsonl',repo_type='dataset',token=os.environ['HF_TOKEN']),'/workspace/st/pool.jsonl')"
echo "== stage A: mel for labeled"; grep -q PREP_DONE /workspace/st/prepA.log 2>/dev/null || python scripts/prep_clap_v2.py --subset /workspace/st/labeled.jsonl --out /workspace/st/mel_labeled --workers 24 > /workspace/st/prepA.log 2>&1
echo "== stage B: teacher"; [ -f /workspace/st/teacher/best/vocal_meta.json ] || python scripts/train_clap_v3.py --data /workspace/st/mel_labeled --out /workspace/st/teacher --batch 48 --bg-frac 0.4 --max-steps 1500 --eval-every 100 --augment > /workspace/st/teacher.log 2>&1
grep -E "EVAL" /workspace/st/teacher.log | tail -1 | cut -c1-300
echo "== stage C: mel for pool"; grep -q PREP_DONE /workspace/st/prepC.log 2>/dev/null || python scripts/prep_clap_v2.py --subset /workspace/st/pool.jsonl --out /workspace/st/mel_pool --workers 24 > /workspace/st/prepC.log 2>&1
python - <<'PY'
# carry the old label into the pool index so the agreement filter can use it
import json; old = {json.loads(l)["uid"]: json.loads(l)["old"] for l in open("/workspace/st/pool.jsonl")}
rows = [json.loads(l) for l in open("/workspace/st/mel_pool/index.jsonl")]
with open("/workspace/st/mel_pool/index.jsonl", "w") as f:
    for r in rows: r["old"] = old.get(r["uid"]); f.write(json.dumps(r) + "\n")
PY
echo "== stage D: pseudo-label"; python scripts/pseudo_label_clap.py --ckpt /workspace/st/teacher/best --labeled /workspace/st/mel_labeled --data /workspace/st/mel_pool --durations /workspace/st/pool.jsonl --out /workspace/st/pseudo_agree.jsonl --require-old-agree > /workspace/st/pseudoD.log 2>&1; tail -1 /workspace/st/pseudoD.log
python scripts/pseudo_label_clap.py --ckpt /workspace/st/teacher/best --labeled /workspace/st/mel_labeled --data /workspace/st/mel_pool --durations /workspace/st/pool.jsonl --out /workspace/st/pseudo_all.jsonl > /workspace/st/pseudoD2.log 2>&1; tail -1 /workspace/st/pseudoD2.log
echo "== stage E: student (labeled + pseudo, shards from both dirs)"; mkdir -p /workspace/st/mel_student
for f in /workspace/st/mel_labeled/shard_*.f16; do ln -sf $f /workspace/st/mel_student/$(basename $f); done
n=$(ls /workspace/st/mel_labeled/shard_*.f16 | wc -l)
python - <<PY
import json, glob, os
n = $n; rows = [json.loads(l) for l in open("/workspace/st/mel_labeled/index.jsonl")]
for p in sorted(glob.glob("/workspace/st/mel_pool/shard_*.f16")):
    i = int(os.path.basename(p)[6:10]); dst = f"/workspace/st/mel_student/shard_{n+i:04d}.f16"
    if not os.path.exists(dst): os.symlink(p, dst)
with open("/workspace/st/mel_student/index.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
for name in ("pseudo_agree", "pseudo_all"):
    with open(f"/workspace/st/{name}_shifted.jsonl", "w") as f:
        for l in open(f"/workspace/st/{name}.jsonl"):
            r = json.loads(l); r["shard"] += n; f.write(json.dumps(r) + "\n")
PY
python scripts/train_clap_v3.py --data /workspace/st/mel_student --out /workspace/st/student_agree --pseudo-index /workspace/st/pseudo_agree_shifted.jsonl --batch 48 --bg-frac 0.4 --max-steps 3000 --eval-every 150 --augment > /workspace/st/student_agree.log 2>&1
grep -E "EVAL" /workspace/st/student_agree.log | tail -1 | cut -c1-300
python scripts/train_clap_v3.py --data /workspace/st/mel_student --out /workspace/st/student_all --pseudo-index /workspace/st/pseudo_all_shifted.jsonl --batch 48 --bg-frac 0.4 --max-steps 3000 --eval-every 150 --augment > /workspace/st/student_all.log 2>&1
grep -E "EVAL" /workspace/st/student_all.log | tail -1 | cut -c1-300
echo "== stage F: push"
python - <<'PY'
import json; from huggingface_hub import HfApi
api = HfApi()
def best(p):
    evs = [json.loads(l) for l in open(p + "/eval.jsonl")]; return max(evs, key=lambda e: (e["best"]["precision"] if e.get("best") else 0) * 0.5 + e["acc3"] * 0.5)
for name in ("teacher", "student_agree", "student_all"):
    b = best(f"/workspace/st/{name}"); print(name, json.dumps(b)[:300])
    api.upload_folder(folder_path=f"/workspace/st/{name}/best", repo_id="aoxo/clap-htsat-unfused-asmr-v2", repo_type="model", path_in_repo=f"v3/{name}", commit_message=f"vocal CLAP v3 {name}: acc3={b['acc3']:.3f} best={b['best']}")
api.upload_file(path_or_fileobj="/workspace/st/pseudo_all.jsonl", path_in_repo="v2/pseudo_labels_all.jsonl", repo_id="aoxo/clap-ft-data", repo_type="dataset", commit_message="CLAP v3 teacher pseudo-labels over pool")
api.upload_file(path_or_fileobj="/workspace/st/pseudo_agree.jsonl", path_in_repo="v2/pseudo_labels_agree.jsonl", repo_id="aoxo/clap-ft-data", repo_type="dataset", commit_message="CLAP v3 teacher pseudo-labels (old-label agreement)")
PY
echo PIPELINE_DONE
