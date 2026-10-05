#!/usr/bin/env python3
"""Let CLAP-ASMR v8 order the crowd site's queue: every clip on the site gets v8's uncertainty (max over labels of
1 - |2P - 1|), which is what the site's need score ranks by, so people hear the clips v8 is least sure about.

A clip that was taken down as confident (working_set.py) but that v8 finds uncertain (>= --flip) becomes 'unsure', so
the next working_set.py --reload can bring it back. People's own clips (controls) and clips listed in --keep (a batch
queued on purpose, e.g. targeted brushing / paper rustling) keep their priority.

  python score_queue_v8.py --ckpt D:/t2a/clap_v8/best --pools D:/t2a/pool D:/t2a/pool_yt ... --humans humans/*.jsonl \\
      --keep D:/t2a/pool_ytdense_all/targeted_brushing_paper.jsonl --env-file D:/t2a/asmrboard.env --out D:/t2a/v8_scores.jsonl
"""
from __future__ import annotations
import argparse, glob, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "sync"))
from train_clap_v8 import FRAMES, HOP, LABELS, N_FFT, load_clip, log
from push_clips import Site, load_env_file


def safe(uid: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pools", type=Path, nargs="+", required=True, help="pool dirs (pool.jsonl + pool_*/clips)")
    ap.add_argument("--humans", type=Path, nargs="*", default=[], help="people's label files: their clips keep their priority")
    ap.add_argument("--keep", type=Path, nargs="*", default=[], help="manifests of clips queued on purpose: left alone")
    ap.add_argument("--flip", type=float, default=0.8, help="v8 uncertainty at which a clip becomes 'unsure'")
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True, help="v8 probabilities per scored clip (jsonl)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import torch
    from transformers import ClapFeatureExtractor, ClapModel
    load_env_file(a.env_file); site = Site()
    skip = {json.loads(l)["uid"] for f in a.humans for l in open(f, encoding="utf-8") if json.loads(l).get("uid")}
    skip |= {json.loads(l)["uid"] for f in a.keep for l in open(f, encoding="utf-8")}
    files, verify = {}, set()
    for d in a.pools:
        for f in glob.glob(str(d / "pool_*" / "clips" / "*.mp3")): files.setdefault(Path(f).stem, f)
    uids = {}
    for d in a.pools:
        p = d / "pool.jsonl"
        if not p.exists(): continue
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); f = files.get(safe(r["uid"]))
            if f and r["uid"] not in skip: uids[r["uid"]] = f
            if r.get("src") == "yt_dense": verify.add(r["uid"])
    cand = list(uids); known = set()
    for i in range(0, len(cand), 500):
        known |= set(site.rpc("admin_known_sources", p_source_uids=cand[i:i + 500]))
    todo = [u for u in cand if u in known]
    log(f"{len(todo)} clips on the site to score ({len(skip)} people's / kept clips left alone)")

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
        try: return u, load_clip(uids[u])
        except Exception: return u, None

    items, flipped, scored = [], 0, 0
    with ThreadPoolExecutor(12) as ex, open(a.out, "w", encoding="utf-8") as fh:
        for i in range(0, len(todo), 64):
            got = [(u, x) for u, x in ex.map(safe_load, todo[i:i + 64]) if x is not None]
            if not got: continue
            P = probs(np.stack([x for _, x in got]))
            for (u, _), p in zip(got, P):
                unc = float(np.max(1 - np.abs(2 * p - 1)))
                item = {"source_uid": u, "uncertainty": round(unc, 4)}
                if unc >= a.flip and u not in verify: item["kind"] = "unsure"; flipped += 1
                items.append(item); scored += 1
                fh.write(json.dumps({"uid": u, "uncertainty": round(unc, 4), "p": {c: round(float(v), 3) for c, v in zip(LABELS, p) if v >= 0.05}}) + "\n")
            if (i // 64) % 50 == 0: log(f"  {scored}/{len(todo)} scored")
    u = np.array([x["uncertainty"] for x in items])
    log(f"v8 uncertainty: median {np.median(u):.2f}, >= 0.8 for {(u >= 0.8).mean():.0%}, <= 0.2 for {(u <= 0.2).mean():.0%}; "
        f"{flipped} clips marked unsure")
    if not a.dry_run:
        for i in range(0, len(items), 500): site.rpc("admin_add_clips", p_clips=items[i:i + 500])
        site.rpc("admin_refresh_progress")
    log(f"QUEUE_SCORED {len(items)} clips" + (" (dry run: nothing written)" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
