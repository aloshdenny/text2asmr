#!/bin/bash
# One-shot status of every text2asmr service, for the hourly check-in. Read-only: it never starts or stops
# anything. Each section prints OK / WARN / FAIL lines so a reader (or the scheduled agent) can act on them.
#
#   scripts/status_all.sh            # run from the Mac (orchestrator)
set -uo pipefail
source ~/.t2a_env
DO=root@139.59.33.163
RS=research@100.86.165.70
T="timeout 60"
command -v timeout >/dev/null || T=""
hr() { printf '\n== %s ==\n' "$1"; }

hr "DO droplet"
ssh -o ConnectTimeout=20 $DO 'bash -s' <<'EOF' 2>&1 || echo "FAIL droplet unreachable"
for u in t2a-ytv t2a-supervisor t2a-runpod-guard t2a-fountain; do
  s=$(systemctl is-active $u); [ "$s" = active ] && echo "OK   $u" || echo "FAIL $u is $s"
done
echo "     disk free $(df -h / | awk 'NR==2{print $4}'), mem avail $(free -m | awk 'NR==2{print $7}') MB"
ok=$(grep -c ": OK " /root/t2a/ytv.log); rej=$(grep -c "REJECT" /root/t2a/ytv.log); blk=$(grep -c "blocked" /root/t2a/ytv.log)
last=$(grep -E ": OK |REJECT|blocked" /root/t2a/ytv.log | tail -1 | cut -c1-110)
echo "     youtube verified: OK=$ok REJECT=$rej blocks=$blk | last: $last"
recent_blk=$(awk -v c="$(date -u -d '-2 hours' '+%F %T')" '/blocked/ && substr($0,2,19) > c' /root/t2a/ytv.log | wc -l)
[ "$recent_blk" -ge 2 ] && echo "WARN $recent_blk YouTube blocks in the last 2 h"
echo "     gemini filter: $(grep -E '/33771' /root/t2a/gclean.log | tail -1 | cut -c1-150)"
echo "     gemini ledger: $(cat /root/t2a/gclean/ledger.json 2>/dev/null)"
grep -q "DONE judged" /root/t2a/gclean.log && echo "OK   gemini filter finished"
echo "     gemini vocal judge: $(grep -E "judged|DONE|calibration|sample|Traceback" /root/t2a/gvocal.log 2>/dev/null | tail -1 | cut -c1-140)"
echo "     supervisor: $(grep -E 'balance' /root/t2a/supervisor.log | tail -1 | cut -c1-120)"
echo "     guard: $(tail -1 /root/t2a/runpod_guard.log | cut -c1-120)"
EOF

hr "RunPod"
python3 - <<'PY' 2>&1 || echo "FAIL runpod api"
import json, os, urllib.request
q = "{ myself { clientBalance pods { name desiredStatus costPerHr runtime { uptimeInSeconds gpus { gpuUtilPercent } } } } }"
r = urllib.request.Request("https://api.runpod.io/graphql?api_key=" + os.environ["RUNPOD_API_KEY"], data=json.dumps({"query": q}).encode(),
                           headers={"Content-Type": "application/json", "User-Agent": "t2a-status/1.0"})
m = json.loads(urllib.request.urlopen(r, timeout=60).read())["data"]["myself"]
bal = m["clientBalance"]; burn = sum(p["costPerHr"] for p in m["pods"] if p["desiredStatus"] == "RUNNING")
print(f"{'WARN' if bal < 8 else 'OK  '} balance ${bal:.2f}, burn ${burn:.2f}/h" + (f", ~{max(0, bal - 2) / burn:.1f} h to the $2 guard floor" if burn else ""))
names = [p["name"] for p in m["pods"]]
for p in m["pods"]:
    rt = p.get("runtime") or {}; g = (rt.get("gpus") or [{}])[0].get("gpuUtilPercent")
    up = (rt.get("uptimeInSeconds") or 0) / 3600
    flag = "WARN" if (g is not None and g < 5 and up > 0.75) else "OK  "
    print(f"{flag} {p['name']:22} {p['desiredStatus']:8} up {up:4.1f} h  gpu {g}%  ${p['costPerHr']}/h")
dups = {n for n in names if names.count(n) > 1}
if dups: print(f"WARN duplicate pod names: {sorted(dups)}")
PY

hr "Research server (RTX 4090, D:\\t2a)"
ssh -o ConnectTimeout=20 $RS "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\t2a\\status.ps1" 2>/dev/null | tr -d '\000' | grep -vE "^\s*$" | cut -c1-200
[ "${PIPESTATUS[0]}" = 0 ] || echo "FAIL research server unreachable"

hr "Hub artefacts"
python3 - <<'PY' 2>&1 || echo "FAIL hub"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
fs = api.list_repo_files("aoxo/asmr-yt-chapters", repo_type="dataset")
aud = sum(f.startswith("audio/") for f in fs); tr = sum(f.startswith("audio/") and f.endswith(".flac.json") for f in fs)
print(f"     asmr-yt-chapters: {aud - tr} audio, {tr} whisper transcripts, {sum(f.startswith('labels/video/') for f in fs)} verified singles, {sum(f.startswith('vad/') for f in fs)} vad maps")
cf = api.list_repo_files("aoxo/clap-ft-data", repo_type="dataset")
print("     clap v7 checkpoints: " + ", ".join(sorted({f.split('/')[1] for f in cf if f.startswith('v7_ckpt/')})))
PY
