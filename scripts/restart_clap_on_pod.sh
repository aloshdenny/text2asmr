#!/bin/bash
set -euo pipefail
python - <<'PYK'
import os, signal, subprocess
out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
for line in out.splitlines():
    cmd = line[line.find(" "):].strip() if " " in line else line
    if "scripts/clap_label_audios2.py" in line or line.endswith("kick_clap.sh") or "/tmp/kick_clap.sh" in line:
        pid = int(line.split()[0])
        if pid != os.getpid():
            try:
                os.kill(pid, signal.SIGTERM)
                print("killed", pid)
            except ProcessLookupError:
                pass
print("kill_done")
PYK
sleep 2
nohup /tmp/kick_clap.sh >> /workspace/clap_label.log 2>&1 &
echo KICKED=$!
sleep 25
ps -eo pid,cmd | python -c 'import sys; [print(l.rstrip()) for l in sys.stdin if "clap_label" in l or "kick_clap" in l]'
echo ---LOG---
tail -n 40 /workspace/clap_label.log
