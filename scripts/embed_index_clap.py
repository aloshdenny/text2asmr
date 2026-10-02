#!/usr/bin/env python3
"""Large-scale weak supervision, step 4a: CLAP embeddings for every training row and every labelled clip.

The fused labels cover ~11k clips; CLAP trains on ~254k index rows. A probe trained on the fused clips' embeddings
carries the fusion to every row (the label model -> end model step of weak supervision), so both need embeddings.
The mel shards total 36 GB and the research server has ~10 GB free, so they stream from the Hub one at a time:
v7_mels/shard_*.f16 file by file (deleted after use), yt_mels/ytmels.tar as one HTTP stream, member by member.
Pool clips (the 6 s windows the labellers heard) are embedded from their audio with the training mel frontend.
Per-shard parts are kept, so a rerun skips finished shards.

  python embed_index_clap.py --pools D:\\t2a\\pool D:\\t2a\\pool_yt D:\\t2a\\pool_spray --out D:\\t2a\\emb
Out: <out>/{indomain,yt,clips}.f16 (N x 512 float16, L2-normalised) + matching .jsonl rows (uid first).
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, tarfile, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_pool_clap import FRAMES, MELS, Scorer, repeatpad
from pool_local_labels import load

REPO = "aoxo/clap-ft-data"


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def embed_array(sc: Scorer, mels: np.ndarray, chunk: int) -> np.ndarray:
    out = [sc.embed_mels(mels[i:i + chunk].astype(np.float32)).half().cpu().numpy() for i in range(0, len(mels), chunk)]
    return np.concatenate(out) if out else np.zeros((0, 512), np.float16)


def finish(out: Path, name: str, rows: list[dict], parts: dict[int, np.ndarray]) -> None:
    """Rows -> their (shard, row) embedding, in index order."""
    E = np.stack([parts[r["shard"]][r["row"]] for r in rows])
    E.tofile(out / f"{name}.f16")
    (out / f"{name}.jsonl").write_text("".join(json.dumps({k: r.get(k) for k in ("uid", "label", "split", "shard", "row", "src")}) + "\n"
                                               for r in rows), encoding="utf-8")
    log(f"{name}: {len(rows)} rows -> {out / (name + '.f16')}")


def indomain(sc: Scorer, a) -> None:
    from huggingface_hub import hf_hub_download
    tmp = a.out / "tmp"; parts_dir = a.out / "parts"; parts_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(hf_hub_download(REPO, "v7_mels/index.jsonl", repo_type="dataset", local_dir=tmp), encoding="utf-8")]
    parts = {}
    for s in sorted({r["shard"] for r in rows}):
        pp = parts_dir / f"indomain_{s:04d}.npy"
        if not pp.exists():
            t0 = time.time()
            f = Path(hf_hub_download(REPO, f"v7_mels/shard_{s:04d}.f16", repo_type="dataset", local_dir=tmp))
            mm = np.memmap(f, dtype=np.float16, mode="r", shape=(os.path.getsize(f) // (FRAMES * MELS * 2), FRAMES, MELS))
            t1 = time.time(); np.save(pp, embed_array(sc, mm, a.chunk)); del mm; f.unlink()
            log(f"indomain shard {s}: download {t1 - t0:.0f} s, embed {time.time() - t1:.0f} s")
        parts[s] = np.load(pp)
    finish(a.out, "indomain", rows, parts)


def youtube(sc: Scorer, a) -> None:
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    parts_dir = a.out / "parts"; parts_dir.mkdir(parents=True, exist_ok=True); idx = None
    r = requests.get(hf_hub_url(REPO, "yt_mels/ytmels.tar", repo_type="dataset"), headers=build_hf_headers(), stream=True, timeout=120)
    r.raise_for_status(); r.raw.decode_content = True
    t0 = time.time()
    with tarfile.open(fileobj=r.raw, mode="r|") as tf:
        for m in tf:
            if not m.isfile(): continue
            base = os.path.basename(m.name)
            if base == "index.jsonl":
                idx = [json.loads(l) for l in tf.extractfile(m).read().decode("utf-8").splitlines() if l.strip()]
            elif re.fullmatch(r"shard_\d+\.f16", base):
                s = int(base[6:-4]); pp = parts_dir / f"yt_{s:04d}.npy"
                data = tf.extractfile(m).read()                       # read even when done: the stream must advance
                if not pp.exists():
                    mels = np.frombuffer(data, dtype=np.float16).reshape(-1, FRAMES, MELS)
                    np.save(pp, embed_array(sc, mels, a.chunk))
                log(f"yt shard {s} ({len(data) / 1e6:.0f} MB) done, {(time.time() - t0) / 60:.1f} min in")
    assert idx is not None, "no index.jsonl in ytmels.tar"
    finish(a.out, "yt", idx, {int(p.stem[3:]): np.load(p) for p in parts_dir.glob("yt_*.npy")})


def clips(sc: Scorer, a) -> None:
    rows, E = [], []
    for pool in a.pools:
        files = {Path(f).stem: f for f in glob.glob(str(pool / "pool_*" / "clips" / "*.mp3"))}
        name = lambda r: re.sub(r"[^A-Za-z0-9_.-]", "_", r["uid"])
        items = [(r, files[name(r)]) for r in map(json.loads, open(pool / "pool.jsonl", encoding="utf-8")) if name(r) in files]
        for i in range(0, len(items), a.chunk):
            b = items[i:i + a.chunk]
            E.append(sc.embed_mels(sc.mels(np.stack([repeatpad(load(f, 48000)) for _, f in b]))).half().cpu().numpy())
            rows += [{"uid": r["uid"], "label": r.get("label"), "src": r.get("src"), "pool": pool.name} for r, _ in b]
        log(f"clips: {pool} -> {len(items)}")
    np.concatenate(E).tofile(a.out / "clips.f16")
    (a.out / "clips.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    log(f"clips: {len(rows)} rows")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="v7_ckpt/stage2_best", help=f"checkpoint dir inside {REPO}, or a local dir")
    ap.add_argument("--pools", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=["clips", "indomain", "yt"])
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ck = Path(a.ckpt)
    if not (ck / "config.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{a.ckpt}/*"], local_dir=a.out / "ckpt")
        ck = a.out / "ckpt" / a.ckpt
    sc = Scorer(ck, dev="cuda")
    for step in a.only: {"clips": clips, "indomain": indomain, "yt": youtube}[step](sc, a)
    log("EMBED_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
