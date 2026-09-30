#!/usr/bin/env python3
"""Read labels out of returned ASMR Ear Check kits (the labeller's "Save labelled copy" files).

  python read_label_html.py kit_alice.html kit_bob.html --key key.jsonl --out labels.jsonl
"""
import argparse, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+", type=Path)
ap.add_argument("--key", type=Path, default=None, help="optional: join the hidden strata back on")
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()
key = {json.loads(l)["id"]: json.loads(l) for l in open(a.key)} if a.key else {}
rows = []
for f in a.files:
    m = re.search(r'<script type="application/json" id="kit-answers">(.*?)</script>', f.read_text(), re.S)
    data = json.loads(m.group(1)) if m else {}
    for cid, v in (data.get("answers") or {}).items():
        rows.append({"file": f.name, "who": data.get("who"), "kit": data.get("kit"), "clip": cid, **v,
                     **({"stratum": key[cid]["stratum"], "uid": key[cid]["uid"]} if cid in key else {})})
    print(f"{f.name}: {data.get('who')!r}, {len(data.get('answers') or {})} labelled, saved {data.get('saved_at')}", file=sys.stderr)
out = "".join(json.dumps(r) + "\n" for r in rows)
(a.out.write_text(out) if a.out else sys.stdout.write(out))
