#!/usr/bin/env python3
"""Re-fuse every vocal verdict we have: human calibration + all Gemini items on the Hub + any local extra item
files (e.g. the targeted v1.1 tag-gap judgments). Writes one fused file for the retagger / CLAP builder.

  python fuse_all_vocal.py --extra D:\\t2a\\gtags\\indomain.jsonl --out D:\\t2a\\fusion\\fused_vocal_all2.jsonl
"""
import argparse, json, subprocess, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--extra", type=Path, nargs="*", default=[])
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--work", type=Path, default=Path("fusion_work"))
a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True)
calib = hf_hub_download("aoxo/clap-ft-data", "judges/gemini_vocal_calib.jsonl", repo_type="dataset", local_dir=str(a.work))
base = hf_hub_download("aoxo/clap-ft-data", "judges/gemini_vocal_all_items.jsonl", repo_type="dataset", local_dir=str(a.work))
items = {}
for f in [Path(base)] + a.extra:
    for l in open(f, encoding="utf-8"):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("pred") is not None and r.get("qwen"): items[r["uid"]] = r
merged = a.work / "items_merged.jsonl"
merged.write_text("".join(json.dumps(r) + "\n" for r in items.values()), encoding="utf-8")
print(f"{len(items)} items", flush=True)
rc = subprocess.call([sys.executable, str(Path(__file__).with_name("fuse_judges.py")), "--calib-jsonl", calib, "--items", str(merged),
                      "--label-field", "qwen", "--extra-classes", "moaning,normal speech", "--out", str(a.out)])
sys.exit(rc)
