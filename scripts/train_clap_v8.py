#!/usr/bin/env python3
"""CLAP-ASMR v8: fine-tune CLAP v7 (stage 2) on what people heard, plus the trigger windows two signals agree on.

People are the ground truth. Three kinds of training rows, each a 6 s clip on disk, decoded on the fly (mel shards
would be 128 KB a row; the clips are ~36 KB mp3s already on this disk), each with a soft target per label and a mask
of the labels that target speaks for:
  human    every clip a person labelled (Ear Check kits + ASMR Board): target = share of its people who ticked the
           label, over the labels their menu offered. Recordings split 60/20/20 train / val / test: val picks the
           checkpoint, test is read once at the end.
  trigger  YouTube trigger windows where Gemini hears the sound the video's title names (rank_for_people.py tier 3):
           target = fused P (fuse_labels.py) for every label.
  replay   in-domain clips whose fused P is decided for every label (each <= --neg or >= --pos): keeps v7's vocal
           knowledge while the triggers move in.
No trigger or replay row comes from a recording in val or test.

Loss: masked BCE on a 20-label head (human rows weighted --human-weight) + CLAP's contrastive loss on one positive
label's prompt per row, so the text tower stays aligned for zero-shot use. --freeze-tower trains the head alone on
frozen v7 embeddings: the baseline v8 has to beat on the same rows.

  python train_clap_v8.py --init D:/t2a/emb/ckpt/v7_ckpt/stage2_best --fused D:/t2a/fused/fused.jsonl \\
      --humans D:/t2a/fused/humans/*.jsonl D:/t2a/fused/crowd_all.jsonl --clips D:/t2a/pool D:/t2a/pool_yt ... \\
      --trigger-manifest D:/t2a/pool_ytdense_all/site_manifest.jsonl --external D:/t2a/gate_ext48/clips.json --out D:/t2a/clap_v8
"""
from __future__ import annotations
import argparse, glob, hashlib, json, math, random, re, sys, threading, time
from collections import defaultdict
from pathlib import Path
from queue import Queue

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_labels import LABELS

SR, MAX_S, FRAMES, MELS, N_FFT, HOP = 48000, 480000, 1001, 64, 1024, 480
OLD_MENU = [l for l in LABELS if l != "spraying"]        # Ear Check kits before spraying was on the menu
EXT_MAP = {"page turning": "paper rustling", "typing": "tapping"}
TRIGGERS = ["tapping", "scratching", "crinkling", "brushing", "liquid", "spraying", "microphone touching", "sticky",
            "fabric rustling", "paper rustling", "cutting"]
PROBES = {"brushing": ["a soft brush moving across a microphone", "bristles sweeping"],
          "liquid": ["pouring liquid", "water sloshing in a bottle"], "spraying": ["a spray bottle spraying mist"],
          "silence / room tone": ["quiet room tone, silence"], "background music": ["soft music playing in the background"],
          "normal speech": ["a person talking in a normal voice"], "something else": ["an unusual ASMR sound"]}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def prompts(c: str) -> list[str]:
    return [f"ASMR {c}, close-mic binaural recording", f"the sound of {c}", f"{c} sounds close to a microphone"] + PROBES.get(c, [])


def rec_of(uid: str) -> str:
    return uid.rsplit(":", 1)[0] if uid.startswith("yt:") else re.sub(r"_\d+$", "", uid)


def safe(uid: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def split_of(rec: str) -> str:
    h = int(hashlib.sha1(rec.encode()).hexdigest(), 16) % 10
    return "train" if h < 6 else "val" if h < 8 else "test"


def load_clip(path: str) -> np.ndarray:
    """mp3 / wav -> 10 s at 48 kHz mono float32, repeated to fill (v7's training input)."""
    import io, os
    import soundfile as sf
    from scipy.signal import resample_poly
    p = os.path.abspath(path)                  # clip names are full source paths: past Windows' 260-char limit
    if os.name == "nt" and not p.startswith("\\\\?\\"): p = "\\\\?\\" + p
    with open(p, "rb") as fh: y, s = sf.read(io.BytesIO(fh.read()), dtype="float32", always_2d=True)
    y = y.mean(1)
    if s != SR:
        g = math.gcd(SR, s); y = resample_poly(y, SR // g, s // g).astype(np.float32)
    y = y[:MAX_S]
    if y.size == 0: raise ValueError("empty")
    if y.size < MAX_S: y = np.pad(np.tile(y, MAX_S // y.size), (0, MAX_S % y.size))
    return y


def build_rows(a) -> dict[str, list[dict]]:
    files = {}
    for d in a.clips:
        for f in glob.glob(str(d / "pool_*" / "clips" / "*.mp3")): files.setdefault(Path(f).stem, f)
    drop = {u.strip() for f in a.exclude for u in open(f, encoding="utf-8") if u.strip()}
    fused = {u: p for u, p in ((json.loads(l)["uid"], json.loads(l)["probs"]) for l in open(a.fused, encoding="utf-8")) if u not in drop}
    # people: one vote per person per clip, over that person's menu
    votes: dict[str, dict[str, tuple[set, set]]] = defaultdict(dict)
    for f in a.humans:
        crowd_file = "crowd" in f.name
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("uid") or r.get("skipped") or r["uid"] in drop: continue
            who = r.get("who") or f.stem
            menu = set(r.get("menu") or (LABELS if crowd_file or who.startswith("crowd:") else OLD_MENU))
            votes[r["uid"]][who] = (set(r["labels"]) & set(LABELS), menu)
    rows = defaultdict(list)
    held = set()
    for uid, people in votes.items():
        f = files.get(safe(uid))
        if not f: continue
        y = np.zeros(len(LABELS), np.float32); m = np.zeros(len(LABELS), np.float32)
        for j, c in enumerate(LABELS):
            judges = [labs for labs, menu in people.values() if c in menu]
            if judges: y[j] = sum(c in labs for labs in judges) / len(judges); m[j] = 1.0
        try: load_clip(f)                      # people's clips are few and all evaluated: drop any unreadable one now
        except Exception: continue
        sp = split_of(rec_of(uid))
        if sp != "train": held.add(rec_of(uid))
        rows[f"human_{sp}"].append({"uid": uid, "path": f, "y": y, "m": m, "w": a.human_weight, "n_people": len(people)})
    full = np.ones(len(LABELS), np.float32)
    for l in open(a.trigger_manifest, encoding="utf-8"):
        r = json.loads(l)
        if (r.get("info") or {}).get("tier") != 3 or r["uid"] not in fused or rec_of(r["uid"]) in held: continue
        rows["trigger"].append({"uid": r["uid"], "path": r["path"], "y": np.array([fused[r["uid"]][c] for c in LABELS], np.float32),
                                "m": full, "w": 1.0})
    trig = {x["uid"] for x in rows["trigger"]}
    for uid, p in fused.items():
        if uid in votes or uid in trig: continue
        f = files.get(safe(uid))
        if not f or rec_of(uid) in held: continue
        v = np.array([p[c] for c in LABELS], np.float32)
        if np.all((v <= a.neg) | (v >= a.pos)): rows["replay"].append({"uid": uid, "path": f, "y": v, "m": full, "w": 1.0})
    for k, v in rows.items(): log(f"rows {k}: {len(v)}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="CLAP v7 checkpoint dir (warm start)")
    ap.add_argument("--base", default="laion/clap-htsat-unfused")
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--humans", type=Path, nargs="+", required=True)
    ap.add_argument("--clips", type=Path, nargs="+", required=True, help="pool dirs: clips under pool_*/clips")
    ap.add_argument("--trigger-manifest", type=Path, required=True)
    ap.add_argument("--external", type=Path, default=None, help="human-labelled trigger clips (clips.json) for the final check")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="uid lists never to train or evaluate on (content_filter.py)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--freeze-tower", action="store_true", help="baseline: train the head on frozen v7 embeddings")
    ap.add_argument("--mix", default="0.25,0.35,0.40", help="batch shares: human, trigger, replay")
    ap.add_argument("--human-weight", type=float, default=3.0)
    ap.add_argument("--pos", type=float, default=0.9); ap.add_argument("--neg", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=48); ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--text-lr", type=float, default=5e-6); ap.add_argument("--clip-weight", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=150); ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import ClapFeatureExtractor, ClapModel, ClapProcessor
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev = "cuda"
    rows = build_rows(a)
    train_sets = [rows["human_train"], rows["trigger"], rows["replay"]]
    mix = np.array([float(x) for x in a.mix.split(",")]); mix = mix / mix.sum()
    for i, s in enumerate(train_sets):
        if not s: mix[i] = 0
    mix = mix / mix.sum()

    proc = ClapProcessor.from_pretrained(a.base)
    model = ClapModel.from_pretrained(str(a.init)).to(dev)
    for mod in model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)): mod.eval(); [p.requires_grad_(False) for p in mod.parameters()]
    if a.freeze_tower:
        for p in model.parameters(): p.requires_grad_(False)
    head = torch.nn.Linear(model.config.projection_dim, len(LABELS)).to(dev)
    fe = ClapFeatureExtractor.from_pretrained(a.base)
    fb = torch.from_numpy(np.asarray(fe.mel_filters_slaney, dtype=np.float32).T).to(dev)
    win = torch.hann_window(N_FFT, periodic=True, device=dev)

    def mels(x: "torch.Tensor") -> "torch.Tensor":
        spec = torch.stft(x, N_FFT, HOP, N_FFT, win, center=True, pad_mode="reflect", return_complex=True)
        mel = torch.einsum("mf,bft->btm", fb, spec.real ** 2 + spec.imag ** 2)
        return (10.0 * torch.log10(torch.clamp(mel, min=1e-10)))[:, :FRAMES, :]

    def augment(m: "torch.Tensor", rng: random.Random) -> "torch.Tensor":
        m = m + torch.empty(m.shape[0], 1, 1, device=dev).uniform_(-6, 6)          # gain jitter (dB)
        for i in range(m.shape[0]):
            f0 = rng.randrange(0, MELS - 6); m[i, :, f0:f0 + rng.randrange(1, 6)] = -100.0
            t0 = rng.randrange(0, FRAMES - 80); m[i, t0:t0 + rng.randrange(1, 80), :] = -100.0
        return m

    def audio_emb(x: "torch.Tensor", train: bool, rng=None) -> "torch.Tensor":
        m = mels(x)
        if train and rng is not None: m = augment(m, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model.get_audio_features(input_features=m.unsqueeze(1), is_longer=torch.zeros(len(x), 1, dtype=torch.bool, device=dev))
        return torch.nn.functional.normalize(e.float(), dim=-1)

    tower = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [{"params": list(head.parameters()), "lr": a.head_lr, "weight_decay": 0.0}]
    if tower:
        groups += [{"params": [p for n, p in tower if not n.startswith("text_")], "lr": a.lr, "weight_decay": 0.01},
                   {"params": [p for n, p in tower if n.startswith("text_")], "lr": a.text_lr, "weight_decay": 0.01}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98)); base = [g["lr"] for g in groups]
    lr_mult = lambda s: s / a.warmup if s < a.warmup else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, a.max_steps - a.warmup))))

    q: Queue = Queue(maxsize=8); stop = threading.Event()

    def feeder(seed: int):
        rng = random.Random(seed)
        while not stop.is_set():
            picks = [train_sets[i][rng.randrange(len(train_sets[i]))] for i in np.random.default_rng(rng.randrange(1 << 30)).choice(3, a.batch, p=mix)]
            xs, keep = [], []
            for r in picks:
                try: xs.append(load_clip(r["path"])); keep.append(r)
                except Exception: continue
            if not keep: continue
            q.put((torch.from_numpy(np.stack(xs)).pin_memory(), keep))
    for i in range(a.workers): threading.Thread(target=feeder, args=(a.seed * 100 + i,), daemon=True).start()

    @torch.no_grad()
    def predict(rs: list[dict]) -> np.ndarray:
        model.eval(); out = []
        for i in range(0, len(rs), 32):
            x = torch.from_numpy(np.stack([load_clip(r["path"]) for r in rs[i:i + 32]])).to(dev)
            out.append(torch.sigmoid(head(audio_emb(x, False) * 10.0)).cpu().numpy())
        model.train()
        return np.concatenate(out) if out else np.zeros((0, len(LABELS)))

    def score(rs: list[dict], P: np.ndarray) -> dict:
        from sklearn.metrics import average_precision_score
        Y = np.stack([r["y"] for r in rs]) >= 0.5; M = np.stack([r["m"] for r in rs]) > 0
        aps = {}
        for j, c in enumerate(LABELS):
            k = M[:, j]
            if Y[k, j].sum() >= 3 and (~Y[k, j]).sum() >= 3: aps[c] = float(average_precision_score(Y[k, j], P[k, j]))
        pred = P >= 0.5
        tp = (pred & Y & M).sum(); fp = (pred & ~Y & M).sum(); fn = (~pred & Y & M).sum()
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        return {"mAP": float(np.mean(list(aps.values()))) if aps else 0.0, "f1": 2 * prec * rec / max(1e-9, prec + rec),
                "precision": prec, "recall": rec, "ap": {c: round(v, 3) for c, v in aps.items()}}

    def evaluate(step: int, split: str = "val") -> float:
        rs = rows[f"human_{split}"]; s = score(rs, predict(rs))
        log(f"EVAL {split} step={step} mAP={s['mAP']:.3f} F1={s['f1']:.3f} P={s['precision']:.2f} R={s['recall']:.2f} "
            f"| {' '.join(f'{c}={v:.2f}' for c, v in sorted(s['ap'].items()))}")
        with open(a.out / "eval.jsonl", "a") as fh: fh.write(json.dumps({"step": step, "split": split, **s}) + "\n")
        return s["mAP"]

    def save():
        d = a.out / "best"; d.mkdir(exist_ok=True)
        if not a.freeze_tower: model.save_pretrained(d); proc.save_pretrained(d)
        torch.save(head.state_dict(), d / "head.pt")
        json.dump({"labels": LABELS, "init": str(a.init), "frozen_tower": a.freeze_tower}, open(d / "v8_meta.json", "w"))

    best, step, t0, run, n = -1.0, 0, time.time(), 0.0, 0
    rng = random.Random(a.seed)
    model.train()
    while step < a.max_steps:
        for g, b in zip(opt.param_groups, base): g["lr"] = b * lr_mult(step)
        x, rs = q.get(); x = x.to(dev, non_blocking=True)
        ae = audio_emb(x, True, rng)
        logits = head(ae * 10.0)
        Y = torch.from_numpy(np.stack([r["y"] for r in rs])).to(dev); M = torch.from_numpy(np.stack([r["m"] for r in rs])).to(dev)
        W = torch.tensor([r["w"] for r in rs], device=dev)[:, None]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y, reduction="none")
        loss = (bce * M * W).sum() / (M * W).sum()
        if a.clip_weight > 0 and not a.freeze_tower:
            # contrastive: each row with a positive label pairs with one of its labels' prompts (rows sharing it are positives)
            pos = [(i, rng.choice([LABELS[j] for j in range(len(LABELS)) if r["y"][j] >= 0.5 and r["m"][j] > 0]))
                   for i, r in enumerate(rs) if any(r["y"][j] >= 0.5 and r["m"][j] > 0 for j in range(len(LABELS)))]
            if len(pos) >= 2:
                idx = torch.tensor([i for i, _ in pos], device=dev); lab = [c for _, c in pos]
                t = proc(text=[rng.choice(prompts(c)) for c in lab], return_tensors="pt", padding=True).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): te = model.get_text_features(**t)
                te = torch.nn.functional.normalize(te.float(), dim=-1); scale = model.logit_scale_a.exp().clamp(max=100)
                same = torch.tensor([[x == y for y in lab] for x in lab], dtype=torch.float32, device=dev)
                la = (scale * ae[idx] @ te.T).log_softmax(-1); lt = (scale * te @ ae[idx].T).log_softmax(-1)
                loss = loss + a.clip_weight * (-(la * same).sum(-1) / same.sum(-1) - (lt * same).sum(-1) / same.sum(-1)).mean() / 2
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0); opt.step()
        step += 1; run += loss.item(); n += 1
        if step % 50 == 0: log(f"step {step}/{a.max_steps} loss={run / n:.4f} {step / (time.time() - t0):.2f} it/s"); run = n = 0
        if step % a.eval_every == 0 or step == a.max_steps:
            m = evaluate(step)
            if m > best: best = m; save(); log(f"saved best (val mAP={m:.3f}) step={step}")
    stop.set()
    # final: the best checkpoint on the untouched test people and on the external human-labelled trigger clips
    d = a.out / "best"
    if not a.freeze_tower: model = ClapModel.from_pretrained(str(d)).to(dev)
    head.load_state_dict(torch.load(d / "head.pt", map_location=dev))
    evaluate(step, "test")
    if a.external and a.external.exists():
        ext = [c for c in json.loads(a.external.read_text()) if EXT_MAP.get(c["label"], c["label"]) in LABELS]
        P = predict([{"path": c["wav"]} for c in ext]); tix = [LABELS.index(t) for t in TRIGGERS]
        hit, top, cnt = defaultdict(int), defaultdict(int), defaultdict(int)
        for c, p in zip(ext, P):
            t = EXT_MAP.get(c["label"], c["label"]); cnt[t] += 1; hit[t] += p[LABELS.index(t)] >= 0.5
            top[t] += TRIGGERS[int(np.argmax(p[tix]))] == t
        log("EXTERNAL " + " ".join(f"{t}: P>.5 {100 * hit[t] / cnt[t]:.0f}% top {100 * top[t] / cnt[t]:.0f}%" for t in cnt)
            + f" | overall top-trigger {100 * sum(top.values()) / max(1, sum(cnt.values())):.0f}%")
    log("TRAIN_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
