#!/usr/bin/env python3
"""A single, self-contained HTML labelling kit: N clips embedded as MP3, labels saved *inside* the file.

Anyone can open it offline, label by ear, and click "Save labelled copy": the browser downloads the same page
with their name and answers embedded as JSON. They send that file back; read it with read_label_html.py.
Progress also autosaves in the browser, so closing the tab loses nothing.

The model's guess for each clip (the stratum) is never written into the file: it would anchor the listener.

  python build_label_html.py --sprites DIR --clips DIR/clips.json --key key.jsonl --exclude done_ids.txt --n 150 --out kit.html
"""
from __future__ import annotations
import argparse, base64, json, random, subprocess, tempfile
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

LABEL_GROUPS = [
    ["Voice", ["breathing", "kissing", "oral sounds", "moaning", "whispering", "normal speech"]],
    ["Triggers", ["tapping", "scratching", "crinkling", "brushing", "liquid", "microphone touching", "sticky",
                  "fabric rustling", "paper rustling", "cutting"]],
    ["Other", ["background music", "silence / room tone", "something else"]],
]
PHYSICAL = set(LABEL_GROUPS[1][1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sprites", type=Path, required=True)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, default=None, help="clip ids already labelled")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--kit-id", default="asmr-ear-check-2")
    ap.add_argument("--template", type=Path, default=Path(__file__).with_name("label_kit_template.html"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    clips = {c["id"]: c for c in json.load(open(a.clips))}
    key = {json.loads(l)["id"]: json.loads(l) for l in open(a.key)}
    done = set(a.exclude.read_text().split()) if a.exclude and a.exclude.exists() else set()
    by = defaultdict(list)
    for cid in clips:
        if cid not in done and cid in key: by[key[cid]["stratum"]].append(cid)
    rng = random.Random(7)
    for v in by.values(): rng.shuffle(v)
    # scarce physical strata first in the round-robin, so a short session still covers them
    order = sorted(by, key=lambda s: (s not in PHYSICAL, len(by[s])))
    picked = [c for grp in zip_longest(*[by[s] for s in order]) for c in grp if c][: a.n]
    rng.shuffle(picked)                                  # listening order hides the strata too

    items, tmp = [], Path(tempfile.mkdtemp())
    for i, cid in enumerate(picked):
        c = clips[cid]; mp3 = tmp / f"{cid}.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(c["t"]), "-t", str(c["dur"]), "-i", str(a.sprites / c["sprite"]),
                        "-ac", "1", "-ar", "24000", "-b:a", "48k", str(mp3)], check=True)
        items.append({"id": cid, "dur": round(c["dur"], 2), "src": "data:audio/mpeg;base64," + base64.b64encode(mp3.read_bytes()).decode()})
    html = a.template.read_text()
    html = html.replace("__KIT_ID__", a.kit_id).replace("__CLIPS__", json.dumps(items)).replace("__GROUPS__", json.dumps(LABEL_GROUPS))
    a.out.write_text(html)
    print(f"{len(items)} clips -> {a.out} ({a.out.stat().st_size / 1e6:.1f} MB); strata: "
          + ", ".join(f"{s}={sum(key[c]['stratum'] == s for c in picked)}" for s in order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
