#!/usr/bin/env python3
"""Large-scale weak supervision, the free labellers: AST (AudioSet), a loudness rule, and CLAP v7, run over the clips
label_pool.py kept on disk. Writes ast.jsonl / energy.jsonl / clap.jsonl next to the LLM votes, keyed by uid, so
fuse_labels.py picks them up. AST's AudioSet classes are mapped onto ours (Whispering, Breathing, Crumpling,
Rustle, Tap, Scratch, Water, Music, Silence, Chewing, ...); a class votes "present" above --ast-thresh.

  python pool_local_labels.py --pool D:\\t2a\\pool [--clap D:\\t2a\\labelcmp\\hf\\v7_ckpt\\stage1_best]
"""
from __future__ import annotations
import argparse, glob, json, re, sys, time
from pathlib import Path

import numpy as np

AST_MAP = {"breathing": ["Breathing", "Gasp", "Sigh", "Pant"], "oral sounds": ["Chewing, mastication", "Biting", "Gargling"],
           "moaning": ["Groan", "Grunt", "Whimper"], "whispering": ["Whispering"],
           "normal speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Conversation", "Narration, monologue"],
           "tapping": ["Tap", "Knock"], "scratching": ["Scratch", "Scrape", "Rub"], "crinkling": ["Crumpling, crinkling"],
           "liquid": ["Water", "Liquid", "Drip", "Pour", "Splash, splatter", "Trickle, dribble", "Slosh", "Squish", "Gurgling", "Stir"],
           "spraying": ["Spray"],
           "fabric rustling": ["Rustle", "Zipper (clothing)"], "paper rustling": ["Rustle", "Crumpling, crinkling", "Tearing", "Writing"],
           "cutting": ["Chopping (food)", "Scissors", "Cutlery, silverware"], "background music": ["Music"], "silence / room tone": ["Silence"],
           "microphone touching": ["Thump, thud", "Rumble"]}


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def load(f: str, sr: int) -> np.ndarray:
    """Decode from bytes read by Python: clip names are full source paths and pass Windows' 260-char limit, which
    libsndfile cannot open but Python can with the \\\\?\\ prefix."""
    import io, os, librosa, soundfile as sf
    p = os.path.abspath(f)
    if os.name == "nt" and not p.startswith("\\\\?\\"): p = "\\\\?\\" + p
    with open(p, "rb") as fh: data = fh.read()
    y, s = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return librosa.resample(y.mean(1), orig_sr=s, target_sr=sr) if s != sr else y.mean(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--clap", type=Path, default=None)
    ap.add_argument("--ast-thresh", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="", help="cuda / cpu (default: cuda if free to use) -- cpu keeps off a busy GPU")
    a = ap.parse_args()
    import torch, librosa
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pool = [json.loads(l) for l in open(a.pool / "pool.jsonl", encoding="utf-8")]
    files = {Path(f).stem: f for f in glob.glob(str(a.pool / "pool_*" / "clips" / "*.mp3"))}
    items = [(p["uid"], files[re.sub(r"[^A-Za-z0-9_.-]", "_", p["uid"])]) for p in pool
             if re.sub(r"[^A-Za-z0-9_.-]", "_", p["uid"]) in files]
    log(f"{len(items)} of {len(pool)} pool clips on disk; device {dev}")
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = ASTFeatureExtractor.from_pretrained(name); m = ASTForAudioClassification.from_pretrained(name).to(dev).eval()
    n2i = {v: k for k, v in m.config.id2label.items()}
    scorer = None
    if a.clap:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from score_pool_clap import Scorer, repeatpad
        scorer = Scorer(a.clap, dev=dev); classes = scorer.classes + ["__bg__"]
    out = {k: open(a.pool / f"{k}.jsonl", "w", encoding="utf-8") for k in (["ast", "energy"] + (["clap"] if scorer else []))}
    for i in range(0, len(items), a.batch):
        b = items[i:i + a.batch]
        wavs = [load(f, 16000) for _, f in b]
        with torch.no_grad():
            p = torch.sigmoid(m(**{k: v.to(dev) for k, v in fe(wavs, sampling_rate=16000, return_tensors="pt").items()}).logits).cpu().numpy()
        for (uid, f), x, pr in zip(b, wavs, p):
            probs = {lab: float(max(pr[n2i[n]] for n in ns if n in n2i)) for lab, ns in AST_MAP.items()}
            out["ast"].write(json.dumps({"uid": uid, "menu": sorted(AST_MAP), "labels": [k for k, v in probs.items() if v > a.ast_thresh],
                                         "probs": {k: round(v, 3) for k, v in probs.items()}}) + "\n")
            h = 1600; db = 20 * np.log10(np.sqrt((x[: len(x) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
            out["energy"].write(json.dumps({"uid": uid, "labels": ["silence / room tone"] if np.percentile(db, 90) < -50 else [],
                                            "p90_db": round(float(np.percentile(db, 90)), 1)}) + "\n")
        if scorer:
            w48 = np.stack([repeatpad(load(f, 48000)) for _, f in b])
            for (uid, _), pc in zip(b, scorer(w48)):
                top = classes[int(pc.argmax())]
                out["clap"].write(json.dumps({"uid": uid, "labels": [] if top == "__bg__" else [top], "p": round(float(pc.max()), 3)}) + "\n")
        if (i // a.batch) % 50 == 0: log(f"  {i + len(b)}/{len(items)}")
    for fh in out.values(): fh.close()
    log("LOCAL_LABELS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
