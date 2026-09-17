#!/usr/bin/env python3
"""Pseudo-label a mel index with a vocal CLAP checkpoint + calibrated 4-way head (targets + background) fit on the labeled set.
Keeps only high-precision predictions (prob >= threshold chosen on held-out for target precision); optional old-label agreement; max event duration."""
import argparse, json, os, numpy as np, torch
from collections import Counter
from pathlib import Path
from transformers import ClapModel
FRAMES, MELS, BG = 1001, 64, "__bg__"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--labeled", type=Path, required=True); ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--target-precision", type=float, default=0.85); ap.add_argument("--require-old-agree", action="store_true"); ap.add_argument("--max-dur", type=float, default=5.0)
ap.add_argument("--durations", type=Path, default=None, help="jsonl with uid,duration (pool file) to enforce --max-dur"); ap.add_argument("--chunk", type=int, default=64)
a = ap.parse_args(); meta = json.load(open(Path(a.ckpt) / "vocal_meta.json")); classes = meta["classes"] + [BG]; cid = {c: i for i, c in enumerate(classes)}
m = ClapModel.from_pretrained(a.ckpt).cuda().eval()
def store(d):
    rows = [json.loads(l) for l in open(d / "index.jsonl") if l.strip()]; mm = {}
    def get(r):
        s = r["shard"]
        if s not in mm:
            p = d / f"shard_{s:04d}.f16"; n = os.path.getsize(p) // (FRAMES * MELS * 2); mm[s] = np.memmap(p, dtype=np.float16, mode="r", shape=(n, FRAMES, MELS))
        return np.asarray(mm[s][r["row"]])
    return rows, get
@torch.no_grad()
def embed(rows, get):
    out = []
    for i in range(0, len(rows), a.chunk):
        ch = rows[i:i + a.chunk]; x = torch.from_numpy(np.stack([get(r) for r in ch]).astype(np.float32)).unsqueeze(1).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = m.get_audio_features(input_features=x, is_longer=torch.zeros(len(ch), 1, dtype=torch.bool, device="cuda"))
        out.append(torch.nn.functional.normalize(e.float(), dim=-1))
    return torch.cat(out)
lrows, lget = store(a.labeled); tr = [r for r in lrows if r["split"] == "train"]; ev = [r for r in lrows if r["split"] == "eval"]
Xtr, Xev = embed(tr, lget), embed(ev, lget); ytr = torch.tensor([cid[r["label"]] for r in tr]).cuda(); yev = torch.tensor([cid[r["label"]] for r in ev]).cuda()
hp = Path(a.ckpt) / "head.pt"
if hp.exists():
    sd = torch.load(hp, map_location="cuda"); W = sd["weight"].T.contiguous() * 10.0; b = sd["bias"]; print("using jointly-trained head", flush=True)
else:
    w = torch.bincount(ytr, minlength=len(classes)).float(); cw = (w.sum() / (len(classes) * w.clamp(min=1))).cuda()
    W = torch.zeros(Xtr.shape[1], len(classes), device="cuda", requires_grad=True); b = torch.zeros(len(classes), device="cuda", requires_grad=True); opt = torch.optim.Adam([W, b], lr=5e-3)
    for _ in range(1500):
        loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr, weight=cw) + 1e-3 * (W ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    W = W.detach(); b = b.detach()
with torch.no_grad():
    pe = torch.softmax(Xev @ W + b, 1); conf, pred = pe.max(1); is_pos = yev < len(classes) - 1
    print(f"head held-out: 4-way acc={((pred == yev).float().mean()).item():.3f} | bg recall={(pred[~is_pos] == cid[BG]).float().mean().item():.3f} | 3-way acc on positives={(pred[is_pos] == yev[is_pos]).float().mean().item():.3f}", flush=True)
    chosen = None
    for t in np.linspace(0.5, 0.99, 50):
        keep = (conf >= t) & (pred != cid[BG]); tp = (keep & (pred == yev)).sum().item(); k = keep.sum().item()
        prec = tp / max(1, k); rec = tp / max(1, is_pos.sum().item())
        if prec >= a.target_precision and (chosen is None or rec > chosen["recall"]): chosen = {"t": float(t), "precision": prec, "recall": rec, "kept": k, "bg_leak": ((keep & ~is_pos).sum().item() / max(1, (~is_pos).sum().item()))}
    if chosen is None:
        t = 0.99; keep = (conf >= t) & (pred != cid[BG]); tp = (keep & (pred == yev)).sum().item(); chosen = {"t": t, "precision": tp / max(1, keep.sum().item()), "recall": tp / max(1, is_pos.sum().item()), "kept": int(keep.sum().item()), "bg_leak": None}
    print("operating point:", chosen, flush=True)
    per = {c: {"prec": ((keep := (conf >= chosen["t"]) & (pred == i)) & (yev == i)).sum().item() / max(1, keep.sum().item()), "n_kept": int(keep.sum().item())} for c, i in cid.items() if c != BG}
    print("per-class at operating point:", per, flush=True)
MAP = {"breathing heavy": "breathing", "breathing close": "breathing"}
dur = {}
if a.durations:
    for l in open(a.durations): r = json.loads(l); dur[r["uid"]] = r["duration"]
prows, pget = store(a.data); kept = []; stats = Counter()
with torch.no_grad():
    for i in range(0, len(prows), 2048):
        ch = prows[i:i + 2048]; pe = torch.softmax(embed(ch, pget) @ W + b, 1); conf, pred = pe.max(1)
        for r, c, p in zip(ch, conf.tolist(), pred.tolist()):
            stats["seen"] += 1; lab = classes[p]
            if lab == BG: stats["bg"] += 1; continue
            if c < chosen["t"]: stats["low_conf"] += 1; continue
            if a.require_old_agree and MAP.get(r.get("old"), r.get("old")) != lab: stats["old_disagree"] += 1; continue
            if dur and dur.get(r["uid"], 0) > a.max_dur: stats["too_long"] += 1; continue
            kept.append({"uid": r["uid"], "shard": r["shard"], "row": r["row"], "label": lab, "text": meta["texts"][lab], "split": "train", "conf": round(c, 4)}); stats[lab] += 1
        if (i // 2048) % 20 == 0: print(f"{i}/{len(prows)} {dict(stats)}", flush=True)
with open(a.out, "w") as f:
    for r in kept: f.write(json.dumps(r) + "\n")
json.dump({"operating_point": chosen, "per_class": per, "stats": dict(stats)}, open(str(a.out) + ".meta.json", "w"))
print("DONE", dict(stats), flush=True)
