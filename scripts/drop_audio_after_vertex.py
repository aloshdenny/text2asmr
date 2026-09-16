#!/usr/bin/env python3
"""After the in-flight Vertex submit wave finishes: drop source audio.

Keeps alignment JSON. Restarts the annotator on round=3 so new collects
delete m4a immediately after clips are cut.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

LOG = Path("/Users/aoxo/vscode/text2asmr/label_tool/gemini_vertex_batch.log")
WLOG = Path("/Users/aoxo/vscode/text2asmr/label_tool/drop_audio_watch.log")
OLD_PID = 7961
AUDIO_SUF = {".m4a", ".mp3", ".wav", ".aac", ".ogg", ".webm"}
ROOTS = [
    Path("/Users/aoxo/t2a"),
    Path("/Users/aoxo/vscode/text2asmr/label_tool/batch_work"),
]


def say(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with WLOG.open("a") as f:
        f.write(line + "\n")


def round2_stats() -> tuple[int, bool]:
    text = LOG.read_text(errors="replace")
    in_r2 = submitted = 0
    poll = False
    for line in text.splitlines():
        if "round=2 " in line:
            in_r2 = 1
        elif "round=3 " in line:
            in_r2 = 0
        if in_r2 and "submitted projects/" in line:
            submitted += 1
        if in_r2 and "poll projects/" in line:
            poll = True
    return submitted, poll


def sweep_audio() -> None:
    n = b = 0
    for root in ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            drop = suf in AUDIO_SUF or (suf == ".flac" and "batch_work" in p.parts)
            if not drop:
                continue
            try:
                b += p.stat().st_size
                p.unlink()
                n += 1
            except OSError as exc:
                say(f"skip {p}: {exc}")
    say(f"deleted_audio_files={n} bytes={b}")


def restart_annotator() -> None:
    try:
        os.kill(OLD_PID, 15)
    except OSError:
        pass
    time.sleep(3)
    env = os.environ.copy()
    t2a = Path.home() / ".t2a_env"
    if t2a.exists():
        for line in t2a.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip("'").strip('"')
    env.update({
        "GEMINI_BACKEND": "vertex",
        "HF_HUB_DISABLE_XET": "1",
        "TRANSCRIBE_BASE": "/Users/aoxo/t2a",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    logf = open(LOG, "a")
    proc = subprocess.Popen(
        [
            "python3", "-u", "scripts/annotate_audios2_gemini_batch.py",
            "--until-empty", "--poll", "--poll-sec", "30",
            "--batch-size", "200", "--max-batches", "80",
            "--collect-workers", "32",
            "--file-list", "label_tool/audios2_file_list.json",
        ],
        cwd="/Users/aoxo/vscode/text2asmr",
        env=env,
        stdout=logf,
        stderr=logf,
        start_new_session=True,
    )
    say(f"restarted pid={proc.pid}")


def main() -> None:
    say("watcher start, waiting for round2 vertex submits to finish")
    while True:
        n, poll = round2_stats()
        if n >= 80 or poll:
            say(f"wave done submitted={n} poll={poll}")
            break
        time.sleep(15)
    sweep_audio()
    say("waiting for round=3 to restart annotator with drop-audio code")
    while True:
        if "round=3 " in LOG.read_text(errors="replace"):
            say("round=3 seen, restarting annotator")
            restart_annotator()
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
