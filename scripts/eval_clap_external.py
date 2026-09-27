#!/usr/bin/env python3
"""Score a CLAP checkpoint on human-labelled clips (aoxo/t2a-eval-external: FSD50K eval + ESC-50).

The v7 training evals use held-out rows of the same weak labels it trained on, so they measure agreement
with noisy labels, not correctness. This is the correctness check: people labelled these clips, and none of
them came from our corpora. Uses score_pool_clap.Scorer, i.e. the exact training mel path and head.

Reports, per human class:
  * classes in the checkpoint's ontology: top-1 accuracy (head, background included as a choice)
  * "other sound": how often the model wrongly names one of its classes (false accept)
  * classes the checkpoint does not know: where they land (should mostly be background)

  python eval_clap_external.py --ckpt D:\\t2a\\ckpt\\v7s2 --out D:\\t2a\\eval\\v7s2_external.json
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_pool_clap import Scorer, repeatpad

HUMAN_TO_V7 = {"tapping": "tapping", "scratching": "scratching", "crinkling": "crinkling", "breathing": "breathing",
               "mouth sounds": "oral sounds", "other sound": "__bg__"}
SR = 48_000


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(row, tmp: Path):
    """Download one clip and decode it to 48 kHz mono float32 (CLAP's rate). None on failure."""
    p = tmp / (row["uid"].replace("/", "_") + ".wav")
    try:
        if not p.exists():
            time.sleep(0.4)                             # shared Hub quota: 8 threads at 0.4 s ~ 20 files/s peak, bursts only
            req = urllib.request.Request(row["url"], headers={"User-Agent": "t2a-eval/1.0",
                                                              **({"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if "huggingface.co" in row["url"] and os.environ.get("HF_TOKEN") else {})})
            p.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-t", "10", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                             capture_output=True, check=True, timeout=120).stdout
        w = np.frombuffer(out, dtype=np.float32)
        return row, (repeatpad(w) if w.size > SR // 4 else None)
    except Exception as e:
        return row, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    rows = [json.loads(l) for l in open(hf_hub_download("aoxo/t2a-eval-external", "external_eval_manifest.jsonl",
                                                        repo_type="dataset"), encoding="utf-8")]
    scorer = Scorer(a.ckpt)
    classes = scorer.classes + ["__bg__"]
    tmp = Path(tempfile.mkdtemp(prefix="t2a_ext_"))
    preds: dict[str, Counter] = defaultdict(Counter)
    n_fail, buf = 0, []

    def run(buf):
        probs = scorer(np.stack([w for _, w in buf]))
        for (r, _), p in zip(buf, probs):
            preds[r["t2a_class"]][classes[int(p.argmax())]] += 1

    with ThreadPoolExecutor(3) as ex:
        for r, w in ex.map(lambda r: fetch(r, tmp), rows):
            if w is None: n_fail += 1; continue
            buf.append((r, w))
            if len(buf) >= a.batch: run(buf); buf = []
    if buf: run(buf)

    report, lines = {}, []
    for h, c in sorted(preds.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values()); target = HUMAN_TO_V7.get(h)
        top = ", ".join(f"{k} {v / n:.0%}" for k, v in c.most_common(3))
        if target == "__bg__":
            fa = 1 - c["__bg__"] / n
            report[h] = {"n": n, "false_accept": round(fa, 3), "top": c.most_common(5)}
            lines.append(f"{h:14} n={n:4d}  false accept {fa:.0%}   ({top})")
        elif target:
            acc = c[target] / n
            report[h] = {"n": n, "v7_class": target, "accuracy": round(acc, 3), "top": c.most_common(5)}
            lines.append(f"{h:14} n={n:4d}  accuracy {acc:.0%} as '{target}'   ({top})")
        else:
            report[h] = {"n": n, "not_in_ontology": True, "top": c.most_common(5)}
            lines.append(f"{h:14} n={n:4d}  not in ontology -> {top}")
    print("\n".join(lines)); log(f"{n_fail} clips could not be fetched/decoded")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"ckpt": str(a.ckpt), "classes": scorer.classes, "fetch_failed": n_fail, "per_class": report}, indent=1))
    log(f"EVAL_DONE -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
