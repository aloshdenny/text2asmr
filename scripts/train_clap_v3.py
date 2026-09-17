#!/usr/bin/env python3
"""CLAP v3 (vocal): 3 target classes + unlabeled background negatives; multi-positive contrastive; threshold-tuned eval; pseudo-labeling."""
from __future__ import annotations
import argparse, json, math, os, random, threading, time
from collections import defaultdict
from pathlib import Path
from queue import Queue
import numpy as np
FRAMES, MELS, BG = 1001, 64, "__bg__"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

class Store:
    def __init__(self, d: Path):
        self.d = d; self.mm = {}; self.rows = [json.loads(l) for l in open(d / "index.jsonl") if l.strip()]
    def get(self, r):
        s = r["shard"]
        if s not in self.mm or r["row"] >= self.mm[s].shape[0]:
            p = self.d / f"shard_{s:04d}.f16"; n = os.path.getsize(p) // (FRAMES * MELS * 2)
            self.mm[s] = np.memmap(p, dtype=np.float16, mode="r", shape=(n, FRAMES, MELS))
        return np.asarray(self.mm[s][r["row"]])

def augment(x: np.ndarray, rng) -> np.ndarray:
    x = x + rng.uniform(-6, 6)                                   # gain jitter (dB)
    f0 = rng.randrange(0, MELS - 6); x[:, f0:f0 + rng.randrange(1, 6)] = -100.0   # freq mask
    t0 = rng.randrange(0, FRAMES - 80); x[t0:t0 + rng.randrange(1, 80), :] = -100.0   # time mask
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="laion/clap-htsat-unfused"); ap.add_argument("--init", default="")
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--bg-frac", type=float, default=0.4)
    ap.add_argument("--max-steps", type=int, default=1500); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--text-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50); ap.add_argument("--eval-every", type=int, default=100); ap.add_argument("--eval-chunk", type=int, default=32)
    ap.add_argument("--augment", action="store_true"); ap.add_argument("--precision", default="bf16"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pseudo-index", type=Path, default=None, help="extra index.jsonl (same shards dir) of pseudo-labeled rows to add to train")
    ap.add_argument("--pseudo-weight", type=float, default=1.0, help="sampling weight multiplier for pseudo rows"); ap.add_argument("--ce-weight", type=float, default=1.0, help="weight of joint 4-way (targets+bg) classification loss")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import ClapModel, ClapProcessor
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev = "cuda"
    store = Store(a.data)
    train = [r for r in store.rows if r["split"] == "train"]; ev = [r for r in store.rows if r["split"] == "eval"]
    if a.pseudo_index:
        ps = [json.loads(l) for l in open(a.pseudo_index) if l.strip()]
        for r in ps: r["split"] = "train"; r["pseudo"] = True
        train += ps; log(f"added {len(ps)} pseudo rows")
    classes = sorted({r["label"] for r in store.rows if r["label"] != BG}); cid = {c: i for i, c in enumerate(classes)}
    texts = {}
    for r in store.rows:
        if r["label"] in cid and r["label"] not in texts and r.get("text"): texts[r["label"]] = r["text"]
    pos = [r for r in train if r["label"] != BG]; bg = [r for r in train if r["label"] == BG]
    log(f"classes={classes} train pos={len(pos)} bg={len(bg)} eval={len(ev)}")
    proc = ClapProcessor.from_pretrained(a.base); model = ClapModel.from_pretrained(a.init or a.base).to(dev)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)): m.eval(); [p.requires_grad_(False) for p in m.parameters()]
    head = torch.nn.Linear(model.config.projection_dim, len(classes) + 1).to(dev)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad] + [("head." + n, p) for n, p in head.named_parameters()]
    def groups(ps, lr): return [{"params": [p for n, p in ps if p.ndim >= 2 and "logit_scale" not in n], "lr": lr, "weight_decay": 0.01}, {"params": [p for n, p in ps if not (p.ndim >= 2 and "logit_scale" not in n)], "lr": lr, "weight_decay": 0.0}]
    tn = [(n, p) for n, p in named if n.startswith("text_")]; on = [(n, p) for n, p in named if not n.startswith("text_")]
    opt = torch.optim.AdamW(groups(on, a.lr) + groups(tn, a.text_lr), betas=(0.9, 0.98)); base_lrs = [a.lr, a.lr, a.text_lr, a.text_lr]
    def lr_mult(s): return s / a.warmup if s < a.warmup else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, a.max_steps - a.warmup))))

    by_class = defaultdict(list)
    for r in pos: by_class[r["label"]].append(r)
    cls_list = list(by_class); w = np.array([len(by_class[c]) for c in cls_list], dtype=np.float64) ** 0.5; w /= w.sum()
    n_bg = int(a.batch * a.bg_frac); n_pos = a.batch - n_bg
    cw = torch.ones(len(classes) + 1, device=dev); cw[-1] = 0.5   # bg is over-represented per batch; damp it
    q: Queue = Queue(maxsize=6); stop = threading.Event()
    def feeder():
        rng = random.Random(random.random())
        while not stop.is_set():
            picks = []
            for i in np.random.choice(len(cls_list), n_pos, p=w):
                rs = by_class[cls_list[i]]; r = rs[rng.randrange(len(rs))]
                if r.get("pseudo") and rng.random() > a.pseudo_weight: r = rs[rng.randrange(len(rs))]
                picks.append(r)
            bpicks = [bg[rng.randrange(len(bg))] for _ in range(n_bg)] if bg else []
            feats = [store.get(r).astype(np.float32) for r in picks + bpicks]
            if a.augment: feats = [augment(f, rng) for f in feats]
            x = torch.from_numpy(np.stack(feats)).unsqueeze(1); labels = torch.tensor([cid[r["label"]] for r in picks] + [len(classes)] * len(bpicks))
            t = proc(text=[rng.choice(texts[r["label"]]) for r in picks], return_tensors="pt", padding=True)
            q.put((x.pin_memory(), {k: v.pin_memory() for k, v in t.items()}, labels))
    for _ in range(3): threading.Thread(target=feeder, daemon=True).start()

    def loss_fn(ae, te, labels, scale):
        ae = torch.nn.functional.normalize(ae, dim=-1); te = torch.nn.functional.normalize(te, dim=-1); P = te.shape[0]
        pos_m = (labels[:, None] == labels[None, :]).float()
        la = scale * ae[:P] @ te.T                          # audio(pos) -> text
        lt = scale * te @ ae.T                              # text -> all audio (bg are negatives)
        pos_t = torch.cat([pos_m, torch.zeros(P, ae.shape[0] - P, device=ae.device)], 1)
        def mp(lg, pm): lp = lg.log_softmax(-1); return -((lp * pm).sum(-1) / pm.sum(-1)).mean()
        return (mp(la, pos_m) + mp(lt, pos_t)) / 2

    @torch.no_grad()
    def class_embs():
        out = []
        for c in classes:
            t = proc(text=texts[c], return_tensors="pt", padding=True).to(dev)
            e = torch.nn.functional.normalize(model.get_text_features(**t), dim=-1).mean(0); out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.stack(out)
    @torch.no_grad()
    def embed(rows):
        outs = []
        for i in range(0, len(rows), a.eval_chunk):
            ch = rows[i:i + a.eval_chunk]; x = torch.from_numpy(np.stack([store.get(r) for r in ch]).astype(np.float32)).unsqueeze(1).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.precision == "bf16")):
                e = model.get_audio_features(input_features=x, is_longer=torch.zeros(len(ch), 1, dtype=torch.bool, device=dev))
            outs.append(torch.nn.functional.normalize(e.float(), dim=-1))
        return torch.cat(outs)
    BG_TEXTS = ["a person talking, normal speech", "someone whispering words softly", "quiet room tone, silence", "moaning voice", "spoken conversation"]
    @torch.no_grad()
    def bg_emb():
        t = proc(text=BG_TEXTS, return_tensors="pt", padding=True).to(dev)
        return torch.nn.functional.normalize(model.get_text_features(**t), dim=-1)
    @torch.no_grad()
    def evaluate(step):
        model.eval(); ce = class_embs(); be = bg_emb(); ae = embed(ev); sims = ae @ ce.T; conf, pred = sims.max(1)
        bgs = (ae @ be.T).max(1).values; conf = conf - bgs   # contrastive score: best class prompt minus best background prompt
        y = torch.tensor([cid.get(r["label"], -1) for r in ev], device=dev); is_pos = y >= 0
        acc3 = (pred[is_pos] == y[is_pos]).float().mean().item()
        rec = {c: (pred[(y == i)] == i).float().mean().item() for c, i in cid.items()}
        # threshold sweep: keep clips with margin (top1 - top2 sim) and top1 sim above t; measure precision on positives vs bg false-accepts
        top2 = sims.topk(2, dim=1).values; margin = top2[:, 0] - top2[:, 1]
        best = None
        for t in np.linspace(-0.3, 0.5, 81):
            for mth in (0.0, 0.02, 0.05, 0.1):
                keep = (conf > t) & (margin > mth)
                tp = ((pred == y) & keep & is_pos).sum().item(); fp_bg = (keep & ~is_pos).sum().item(); fp_pos = (keep & is_pos & (pred != y)).sum().item()
                prec = tp / max(1, tp + fp_bg + fp_pos); recall = tp / max(1, is_pos.sum().item()); bg_pass = fp_bg / max(1, (~is_pos).sum().item())
                if recall >= 0.5 and (best is None or prec > best["precision"]): best = {"t": round(float(t), 3), "margin": mth, "precision": prec, "recall": recall, "bg_pass": bg_pass, "kept": int(keep.sum().item())}
        hp = torch.softmax(head(ae * 10.0), 1); hconf, hpred = hp.max(1); ybg = torch.where(is_pos, y, torch.full_like(y, len(classes)))
        h4 = (hpred == ybg).float().mean().item(); bg_rec = (hpred[~is_pos] == len(classes)).float().mean().item(); hbest = None
        for t in np.linspace(0.5, 0.99, 50):
            keep = (hconf >= t) & (hpred != len(classes)); tp = (keep & (hpred == ybg)).sum().item(); k = keep.sum().item()
            prec = tp / max(1, k); recall = tp / max(1, is_pos.sum().item())
            if prec >= 0.85 and (hbest is None or recall > hbest["recall"]): hbest = {"t": round(float(t), 3), "precision": round(prec, 3), "recall": round(recall, 3), "kept": k}
        log(f"HEAD step={step} acc4={h4:.3f} bg_recall={bg_rec:.3f} prec>=.85 point={hbest}")
        acos = (ae[:256] @ ae[:256].T).mean().item()
        rec_s = " ".join(f"{c}={v:.2f}" for c, v in rec.items())
        log(f"EVAL step={step} acc3={acc3:.3f} {rec_s} audio_cos={acos:.3f} | best@recall>=.5: {best}")
        (a.out / "eval.jsonl").open("a").write(json.dumps({"step": step, "acc3": acc3, "recall": rec, "best": best, "head": {"acc4": h4, "bg_recall": bg_rec, "p85": hbest}}) + "\n"); model.train()
        return (hbest["recall"] if hbest else 0.0) + 0.1 * h4

    model.train(); step = 0; t0 = time.time(); run = 0.0; n = 0; best_score = -1
    while step < a.max_steps:
        for g, b in zip(opt.param_groups, base_lrs): g["lr"] = b * lr_mult(step)
        x, t, labels = q.get(); x = x.to(dev, non_blocking=True); labels = labels.to(dev); t = {k: v.to(dev, non_blocking=True) for k, v in t.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.precision == "bf16")):
            ae = model.get_audio_features(input_features=x, is_longer=torch.zeros(len(x), 1, dtype=torch.bool, device=dev)); te = model.get_text_features(**t)
        scale = model.logit_scale_a.exp().clamp(max=100)
        P = te.shape[0]; loss = loss_fn(ae.float(), te.float(), labels[:P], scale)
        if a.ce_weight > 0:
            logits = head(torch.nn.functional.normalize(ae.float(), dim=-1) * 10.0)
            loss = loss + a.ce_weight * torch.nn.functional.cross_entropy(logits, labels, weight=cw)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1; run += loss.item(); n += 1
        if step % 50 == 0: log(f"step {step}/{a.max_steps} loss={run/n:.4f} {step/(time.time()-t0):.2f} it/s"); run = 0.0; n = 0
        if step % a.eval_every == 0:
            sc = evaluate(step)
            if sc > best_score:
                best_score = sc; model.save_pretrained(a.out / "best"); proc.save_pretrained(a.out / "best"); torch.save(head.state_dict(), a.out / "best" / "head.pt"); log(f"saved best (score={sc:.3f}) step={step}")
    sc = evaluate(step)
    if sc > best_score: model.save_pretrained(a.out / "best"); proc.save_pretrained(a.out / "best"); torch.save(head.state_dict(), a.out / "best" / "head.pt")
    json.dump({"classes": classes, "texts": texts, "bg_texts": BG_TEXTS}, open(a.out / "best" / "vocal_meta.json", "w")); stop.set(); log("TRAIN_DONE")

if __name__ == "__main__": main()
