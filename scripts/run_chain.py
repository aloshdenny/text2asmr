#!/usr/bin/env python3
"""Run pipeline stages in order, stopping at the first failure; a stage whose marker file exists is skipped,
so relaunching the chain after a crash resumes at the unfinished stage.

  python run_chain.py --stage "marker1::python a.py ..." --stage "marker2::python b.py ..."
"""
import argparse, shlex, subprocess, sys, time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", action="append", required=True)
    for spec in ap.parse_args().stage:
        marker, cmd = spec.split("::", 1)
        if Path(marker).exists():
            print(f"[chain] skip (done): {cmd[:100]}", flush=True); continue
        print(f"[chain] {time.strftime('%F %T')} START {cmd[:160]}", flush=True)
        rc = subprocess.call(shlex.split(cmd, posix=False))
        print(f"[chain] {time.strftime('%F %T')} rc={rc}", flush=True)
        if rc != 0: return rc
        Path(marker).write_text(time.strftime("%F %T"))
    print("[chain] CHAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
