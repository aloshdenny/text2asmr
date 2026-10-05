#!/usr/bin/env python3
"""Real-or-AI clips for ASMR Board's interludes: real whispered speech, and T2A cloning the same speakers.

Real voices come from Expresso (Meta AI; read speech recorded for expressive speech synthesis, CC BY-NC 4.0), whisper
style. Every voice appears both real and cloned -- a voice's reference lines, real lines and cloned lines never
overlap -- so recognising a voice says nothing about the answer. Both sides get the same finishing (24 kHz mono,
edges trimmed at the same threshold, loudness matched, the room tone between words lifted to the voice's real
level, 64 kb/s mp3 without metadata), so nothing in the file does either: the generator's near-silent gaps
(12 dB under the real recordings' room tone) were an obvious tell on headphones. Only voices recorded for synthesis research are cloned here: never a creator's.

  python rai_items.py prepare --out D:/t2a/rai [--per-voice 30]   # one shard at a time: refs, real clips, gen.jsonl
  python generate_t2a_v1.py --batch D:/t2a/rai/gen.jsonl --adapter D:/t2a/t3dpo/best --device cuda --gate 0
  python rai_items.py finish --out D:/t2a/rai                       # -> final/*.mp3 + manifest.jsonl (sync/push_rai.py)
"""
from __future__ import annotations
import argparse, hashlib, io, json, shutil, subprocess, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = "ylacombe/expresso"
SR = 24000
MODEL = "t2a-v1.2"


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def mono24(data: bytes) -> np.ndarray:
    import librosa
    y, s = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    y = y.mean(1)
    return librosa.resample(y, orig_sr=s, target_sr=SR) if s != SR else y


def order(uid: str) -> str:
    return hashlib.sha1(uid.encode()).hexdigest()               # a fixed shuffle, independent of shard order


def prepare(a) -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    shards = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith("read/") and f.endswith(".parquet"))
    cand = a.out / "candidates"; cand.mkdir(parents=True, exist_ok=True)
    tmp = a.out / "shard"
    rows = []
    for f in shards:
        local = Path(hf_hub_download(REPO, f, repo_type="dataset", local_dir=tmp))
        t = pq.read_table(local).to_pylist()
        kept = 0
        for r in t:
            if r["style"] != "whisper": continue
            y = mono24(r["audio"]["bytes"])
            dur = len(y) / SR
            words = len(r["text"].split())
            if not (2.5 <= dur <= 9.0 and words >= 5): continue
            path = cand / f"{r['id']}.wav"
            sf.write(path, y, SR, subtype="PCM_16")
            rows.append({"id": r["id"], "voice": r["speaker_id"], "text": r["text"].strip(), "dur": round(dur, 2), "path": str(path)})
            kept += 1
        shutil.rmtree(tmp, ignore_errors=True)                  # keep the disk free: one shard at a time
        log(f"{f}: {len(t)} rows, {kept} whisper lines kept")
    by = defaultdict(list)
    for r in rows: by[r["voice"]].append(r)
    (a.out / "refs").mkdir(exist_ok=True); (a.out / "real").mkdir(exist_ok=True)
    gen, real = [], []
    for v, rs in sorted(by.items()):
        rs.sort(key=lambda r: order(r["id"]))
        refs, total = [], 0.0
        while rs and total < 9.0:                                # ~9-12 s reference from lines used nowhere else
            r = rs.pop(0); refs.append(r); total += r["dur"]
        gap = np.zeros(int(0.3 * SR), np.float32)
        ref = np.concatenate([x for r in refs for x in (sf.read(r["path"], dtype="float32")[0], gap)])
        ref_path = a.out / "refs" / f"{v}.wav"; sf.write(ref_path, ref, SR, subtype="PCM_16")
        n = min(a.per_voice, len(rs) // 2)
        for r in rs[:n]:
            dst = a.out / "real" / f"{r['id']}.wav"; shutil.copy(r["path"], dst)
            real.append({**r, "path": str(dst)})
        for r in rs[n:2 * n]:
            gen.append({"ref": str(ref_path), "script": r["text"], "out": str(a.out / "gen" / f"{r['id']}.wav"),
                        "id": r["id"], "voice": v})
        log(f"voice {v}: ref {total:.1f} s from {len(refs)} lines, {n} real + {n} to clone (of {len(rs) + len(refs)} lines)")
    shutil.rmtree(cand, ignore_errors=True)
    (a.out / "real.jsonl").write_text("".join(json.dumps(r) + "\n" for r in real), encoding="utf-8")
    (a.out / "gen.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gen), encoding="utf-8")
    log(f"PREPARED {len(real)} real, {len(gen)} to generate -> {a.out / 'gen.jsonl'}")
    return 0


def finished(y: np.ndarray, pad_s: float = 0.2) -> np.ndarray | None:
    """Trim both edges where the level is 40 dB under the loudest 20 ms (keeping pad_s), match loudness of the
    active part to -24 dBFS RMS, keep peaks under -1 dBFS. The same for real and generated clips."""
    hop = SR // 50
    n = len(y) // hop
    if n < 10: return None
    db = 20 * np.log10(np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    on = np.where(db > db.max() - 40)[0]
    s, e = max(0, on[0] * hop - int(pad_s * SR)), min(len(y), (on[-1] + 1) * hop + int(pad_s * SR))
    y = y[s:e]
    act = db[on[0]:on[-1] + 1]
    rms = np.sqrt(np.mean(10 ** (act[act > db.max() - 30] / 10)))
    y = y * (10 ** (-24 / 20) / max(rms, 1e-6))
    peak = np.abs(y).max()
    return y * (10 ** (-1 / 20) / peak) if peak > 10 ** (-1 / 20) else y


def frame_db(y: np.ndarray, hop: int = SR // 50) -> np.ndarray:
    n = len(y) // hop
    return 20 * np.log10(np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)


def room_bed(clips: list[np.ndarray], hop: int = SR // 50) -> np.ndarray:
    """A voice's room tone: the quietest 15% of 20 ms frames of its real recordings, joined end to end."""
    parts = []
    for y in clips:
        db = frame_db(y, hop)
        for i in np.where(db <= np.percentile(db, 15))[0]: parts.append(y[i * hop:(i + 1) * hop])
    return np.concatenate(parts) if parts else np.zeros(SR, np.float32)


def lift_floor(y: np.ndarray, bed: np.ndarray, target_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add room tone until the clip's quiet (10th-percentile 20 ms frame) is within 0.5 dB of target_db; louder floors
    stay as they are. Room tone is not steady noise, so one top-up undershoots: a few rounds converge."""
    if not len(bed): return y
    out = y.astype(np.float32)
    for _ in range(6):
        have = np.percentile(frame_db(out), 10)
        if have >= target_db - 0.5: break
        need = 10 ** (target_db / 10) - 10 ** (have / 10)
        tone = np.tile(bed, int(np.ceil(len(out) / len(bed))) + 1)
        start = int(rng.integers(0, len(bed))); tone = tone[start:start + len(out)]
        out = out + (tone * np.sqrt(need / max(np.mean(tone ** 2), 1e-12))).astype(np.float32)
    peak = np.abs(out).max()
    return out * (10 ** (-1 / 20) / peak) if peak > 10 ** (-1 / 20) else out


def to_mp3(y: np.ndarray, dst: Path) -> None:
    buf = io.BytesIO(); sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "wav", "-i", "-", "-ac", "1", "-ar", str(SR), "-b:a", "64k",
                    "-map_metadata", "-1", "-id3v2_version", "0", str(dst)],
                   input=buf.getvalue(), check=True)


def finish(a) -> int:
    import librosa
    out = a.out / "final"; out.mkdir(exist_ok=True)
    real = [json.loads(l) for l in open(a.out / "real.jsonl", encoding="utf-8")]
    gen = [json.loads(l) for l in open(a.out / "gen.jsonl", encoding="utf-8")]
    items = defaultdict(lambda: {"real": [], "ai": []})
    for r in real:
        y = finished(sf.read(r["path"], dtype="float32")[0])
        if y is not None: items[r["voice"]]["real"].append((r, y))
    for g in gen:
        if not Path(g["out"]).exists(): continue
        y, s = sf.read(g["out"], dtype="float32", always_2d=True)
        y = y.mean(1); y = librosa.resample(y, orig_sr=s, target_sr=SR) if s != SR else y
        y = finished(y)
        if y is None or not 1.5 <= len(y) / SR <= 12: continue   # a failed take: too short or ran on
        items[g["voice"]]["ai"].append((g, y))
    rows, rng = [], np.random.default_rng(0)
    for v, d in sorted(items.items()):
        n = min(len(d["real"]), len(d["ai"]))                   # as many real as AI per voice: no base-rate tell
        if not n: continue
        bed = room_bed([y for _, y in d["real"]])
        target = float(np.median([np.percentile(frame_db(y), 10) for _, y in d["real"]]))
        for kind in ("real", "ai"):
            for r, y in d[kind][:n]:
                y = lift_floor(y, bed, target, rng)             # same rule for both sides
                dst = out / f"{r['id']}_{kind}.mp3"; to_mp3(y, dst)
                rows.append({"file": str(dst), "is_ai": kind == "ai", "voice": v, "model": MODEL if kind == "ai" else None,
                             "source": r["id"], "dur": round(len(y) / SR, 2)})
        log(f"voice {v}: {n} real + {n} AI")
    (a.out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    dr = [r["dur"] for r in rows if not r["is_ai"]]; da = [r["dur"] for r in rows if r["is_ai"]]
    log(f"FINISHED {len(rows)} items; median length real {np.median(dr) if dr else 0:.1f} s, AI {np.median(da) if da else 0:.1f} s "
        f"-> {a.out / 'manifest.jsonl'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "finish"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-voice", type=int, default=30)
    a = ap.parse_args()
    return prepare(a) if a.step == "prepare" else finish(a)


if __name__ == "__main__":
    raise SystemExit(main())
