#!/usr/bin/env python3
"""CLAP-ASMR v8.x as a free labeller on a pool's clips: its 20-label head's probabilities, and the labels at or above
--thresh, written to <pool>/clap8.jsonl next to the LLM judges' files (merge_votes.py reads it as "clap8").

  python pool_clap8_labels.py --pool D:/t2a/pool_ytdense2 --ckpt D:/t2a/clap_v82/best
"""
from __future__ import annotations
import argparse, glob, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_clap_v8 import FRAMES, HOP, LABELS, N_FFT, load_clip, log


def safe(uid: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()
    import torch
    from transformers import ClapFeatureExtractor, ClapModel
    files = {Path(f).stem: f for f in glob.glob(str(a.pool / "pool_*" / "clips" / "*.mp3"))}
    uids = [r["uid"] for r in map(json.loads, open(a.pool / "pool.jsonl", encoding="utf-8")) if safe(r["uid"]) in files]
    log(f"{len(uids)} clips on disk in {a.pool}")
    dev = "cuda"
    model = ClapModel.from_pretrained(str(a.ckpt)).to(dev).eval()
    head = torch.nn.Linear(model.config.projection_dim, len(LABELS)).to(dev)
    head.load_state_dict(torch.load(a.ckpt / "head.pt", map_location=dev)); head.eval()
    fe = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
    fb = torch.from_numpy(np.asarray(fe.mel_filters_slaney, dtype=np.float32).T).to(dev)
    win = torch.hann_window(N_FFT, periodic=True, device=dev)

    def probs(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(x).to(dev)
            spec = torch.stft(t, N_FFT, HOP, N_FFT, win, center=True, pad_mode="reflect", return_complex=True)
            m = (10.0 * torch.log10(torch.clamp(torch.einsum("mf,bft->btm", fb, spec.real ** 2 + spec.imag ** 2), min=1e-10)))[:, :FRAMES, :]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = model.get_audio_features(input_features=m.unsqueeze(1), is_longer=torch.zeros(len(x), 1, dtype=torch.bool, device=dev))
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            return torch.sigmoid(head(e * 10.0)).cpu().numpy()

    def safe_load(u: str):
        try: return u, load_clip(files[safe(u)])
        except Exception: return u, None

    n = 0
    with ThreadPoolExecutor(12) as ex, open(a.pool / "clap8.jsonl", "w", encoding="utf-8") as fh:
        for i in range(0, len(uids), 64):
            got = [(u, x) for u, x in ex.map(safe_load, uids[i:i + 64]) if x is not None]
            if not got: continue
            for (u, _), p in zip(got, probs(np.stack([x for _, x in got]))):
                fh.write(json.dumps({"uid": u, "labels": [c for c, v in zip(LABELS, p) if v >= a.thresh], "menu": LABELS,
                                     "probs": {c: round(float(v), 3) for c, v in zip(LABELS, p)}}) + "\n"); n += 1
            if (i // 64) % 40 == 0: log(f"  {n}/{len(uids)}")
    log(f"CLAP8_DONE {n} clips -> {a.pool / 'clap8.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
