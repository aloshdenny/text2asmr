#!/usr/bin/env python3
"""Score CLAP probes (probe_fused.py) on human-labelled trigger clips outside our corpus: the FSD50K / ESC-50 clips
class_gate.py --stage cutext cut (clips.json: uid, label, wav). Our people mostly heard vocal clips, so this is the
check that a probe hears real tapping, brushing, liquid... For each class: how often the probe's own label for the
clip (P > 0.5) and its single most likely trigger match what the person heard.

  python probe_external_eval.py --clips D:/t2a/gate_ext/clips.json --ckpt D:/t2a/emb/ckpt/v7_ckpt/stage2_best \\
      --probes D:/t2a/probe_v73/probe.pt D:/t2a/probe_v75/probe.pt
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_labels import LABELS
from pool_local_labels import load
from score_pool_clap import Scorer, repeatpad

MAP = {"page turning": "paper rustling", "typing": "tapping"}
TRIGGERS = ["tapping", "scratching", "crinkling", "brushing", "liquid", "spraying", "microphone touching", "sticky",
            "fabric rustling", "paper rustling", "cutting"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--probes", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    clips = [c for c in json.loads(a.clips.read_text()) if MAP.get(c["label"], c["label"]) in LABELS]
    sc = Scorer(a.ckpt, dev="cuda")
    E = []
    for i in range(0, len(clips), 32):
        w = np.stack([repeatpad(load(c["wav"], 48000)) for c in clips[i:i + 32]])
        E.append(sc.embed_mels(sc.mels(w)).float().cpu().numpy())
    E = torch.tensor(np.concatenate(E))
    tix = [LABELS.index(t) for t in TRIGGERS]
    for p in a.probes:
        lin = torch.nn.Linear(E.shape[1], len(LABELS)); lin.load_state_dict(torch.load(p, map_location="cpu"))
        with torch.no_grad(): P = torch.sigmoid(lin(E)).numpy()
        hit, top, n = defaultdict(int), defaultdict(int), defaultdict(int)
        for c, pr in zip(clips, P):
            t = MAP.get(c["label"], c["label"]); j = LABELS.index(t)
            n[t] += 1; hit[t] += pr[j] > 0.5; top[t] += TRIGGERS[int(np.argmax(pr[tix]))] == t
        print(f"== {p}")
        print(f"  {'class':16}{'n':>4}{'P>0.5':>8}{'top trigger':>13}")
        for t in sorted(n, key=lambda x: -n[x]):
            print(f"  {t:16}{n[t]:4}{100 * hit[t] / n[t]:7.0f}%{100 * top[t] / n[t]:12.0f}%")
        print(f"  overall top-trigger accuracy {100 * sum(top.values()) / sum(n.values()):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
