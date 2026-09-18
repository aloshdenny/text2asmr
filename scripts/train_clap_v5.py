#!/usr/bin/env python3
"""CLAP v5: multiple-instance training. YouTube chapters are bags of windows (top-k mean over windows must match the chapter label);
Pro-labeled vocal clips are single-window bags; whisper/talk chapters and Pro rejects are background bags (mean-pooled).
Eval: chapter-level accuracy on held-out videos + window-level on Pro held-out."""
from __future__ import annotations
import argparse, json, math, os, random, threading, time
from collections import Counter, defaultdict
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
            p = self.d / f"shard_{s:04d}.f16"; n = os.path.getsize(p) // (FRAMES * MELS * 2); self.mm[s] = np.memmap(p, dtype=np.float16, mode="r", shape=(n, FRAMES, MELS))
        return np.asarray(self.mm[s][r["row"]])
def augment(x, rng):
    x = x + rng.uniform(-6, 6); f0 = rng.randrange(0, MELS - 6); x[:, f0:f0 + rng.randrange(1, 6)] = -100.0; t0 = rng.randrange(0, FRAMES - 80); x[t0:t0 + rng.randrange(1, 80), :] = -100.0; return x
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--base", default="laion/clap-htsat-unfused")
    ap.add_argument("--bags", type=int, default=12); ap.add_argument("--k", type=int, default=6, help="windows sampled per YT bag"); ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--contrast-weight", type=float, default=0.5); ap.add_argument("--augment", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import ClapModel, ClapProcessor
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev = "cuda"
    store = Store(a.data)
    # bags: YT -> (source, chapter); Pro -> one per clip
    bags = defaultdict(list)
    for i, r in enumerate(store.rows):
        key = (r["source"], r["chapter"]) if r.get("chapter") else (r.get("source") or r["uid"], r["uid"])
        bags[key].append(i)
    bag_list = [(k, v, store.rows[v[0]]["label"], store.rows[v[0]]["split"], bool(store.rows[v[0]].get("chapter"))) for k, v in bags.items()]
    classes = sorted({b[2] for b in bag_list if b[2] != BG}); cid = {c: i for i, c in enumerate(classes)}; cid[BG] = len(classes)
    texts = {}
    for r in store.rows:
        if r["label"] in cid and r["label"] not in texts and r.get("text"): texts[r["label"]] = r["text"]
    tr = [b for b in bag_list if b[3] == "train"]; ev_yt = [b for b in bag_list if b[3] == "eval" and b[4]]; ev_pro = [i for i, r in enumerate(store.rows) if r["split"] == "eval" and not r.get("chapter")]
    log(f"classes={classes} train bags={len(tr)} (yt={sum(b[4] for b in tr)}) eval yt bags={len(ev_yt)} eval pro clips={len(ev_pro)}")
    log(f"train bag labels: {Counter(b[2] for b in tr).most_common()}")
    proc = ClapProcessor.from_pretrained(a.base); model = ClapModel.from_pretrained(a.base).to(dev)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)): m.eval(); [p.requires_grad_(False) for p in m.parameters()]
    head = torch.nn.Linear(model.config.projection_dim, len(classes) + 1).to(dev)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    def lr_mult(s): return s / a.warmup if s < a.warmup else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, a.max_steps - a.warmup))))
    counts = Counter(b[2] for b in tr); cw = torch.tensor([1.0 / math.sqrt(counts.get(c, 1)) for c in classes] + [1.0 / math.sqrt(counts.get(BG, 1))], device=dev); cw = cw / cw.mean()
    by_label = defaultdict(list)
    for b in tr: by_label[b[2]].append(b)
    labs = list(by_label); lw = np.array([len(by_label[l]) for l in labs], dtype=np.float64) ** 0.5; lw /= lw.sum()
    q: Queue = Queue(maxsize=4); stop = threading.Event()
    def feeder():
        rng = random.Random(random.random())
        while not stop.is_set():
            picks = [by_label[labs[i]][rng.randrange(len(by_label[labs[i]]))] for i in np.random.choice(len(labs), a.bags, p=lw)]
            feats, sizes, labels = [], [], []
            for _, idx, lab, _, is_yt in picks:
                sel = rng.sample(idx, min(a.k, len(idx))) if is_yt else idx[:1]
                for i in sel:
                    f = store.get(store.rows[i]).astype(np.float32); feats.append(augment(f, rng) if a.augment else f)
                sizes.append(len(sel)); labels.append(cid[lab])
            x = torch.from_numpy(np.stack(feats)).unsqueeze(1); q.put((x.pin_memory(), sizes, torch.tensor(labels)))
    for _ in range(3): threading.Thread(target=feeder, daemon=True).start()
    def bag_logits(L, sizes, labels):
        out, o = [], 0
        for n, y in zip(sizes, labels.tolist()):
            Ls = L[o:o + n]; o += n
            out.append(Ls.mean(0) if (y == cid[BG] or n == 1) else Ls.topk(min(a.topk, n), dim=0).values.mean(0))
        return torch.stack(out)
    @torch.no_grad()
    def class_embs():
        out = []
        for c in classes:
            t = proc(text=texts[c], return_tensors="pt", padding=True).to(dev); e = torch.nn.functional.normalize(model.get_text_features(**t), dim=-1).mean(0); out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.stack(out)
    @torch.no_grad()
    def embed_rows(idxs, chunk=48):
        outs = []
        for i in range(0, len(idxs), chunk):
            x = torch.from_numpy(np.stack([store.get(store.rows[j]) for j in idxs[i:i + chunk]]).astype(np.float32)).unsqueeze(1).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = model.get_audio_features(input_features=x, is_longer=torch.zeros(len(x), 1, dtype=torch.bool, device=dev))
            outs.append(torch.nn.functional.normalize(e.float(), dim=-1))
        return torch.cat(outs)
    @torch.no_grad()
    def evaluate(step):
        model.eval(); per = defaultdict(lambda: [0, 0]); correct = 0
        for _, idx, lab, _, _ in ev_yt:
            L = head(embed_rows(idx[:40]) * 10.0); pred = (L.mean(0) if lab == BG else L.topk(min(a.topk, len(idx)), dim=0).values.mean(0)).argmax().item()
            # chapter-level: score bag with top-k for every class column, argmax over classes
            Lk = torch.stack([L[:, c].topk(min(a.topk, len(idx))).values.mean() for c in range(L.shape[1])]); pred = Lk.argmax().item()
            per[lab][1] += 1; per[lab][0] += (pred == cid[lab]); correct += (pred == cid[lab])
        ch_acc = correct / max(1, len(ev_yt)); rec = {c: round(v[0] / v[1], 2) for c, v in per.items() if v[1] >= 3}; macro = np.mean([v for v in rec.values()]) if rec else 0
        pro_acc = None
        if ev_pro:
            L = head(embed_rows(ev_pro) * 10.0); y = torch.tensor([cid[store.rows[i]["label"]] for i in ev_pro], device=dev); pro_acc = (L.argmax(1) == y).float().mean().item()
        log(f"EVAL step={step} yt_chapter_acc={ch_acc:.3f} macro={macro:.3f} pro_window_acc={pro_acc} per_class={rec}")
        (a.out / "eval.jsonl").open("a").write(json.dumps({"step": step, "yt_chapter_acc": ch_acc, "yt_macro": float(macro), "pro_window_acc": pro_acc, "recall": rec}) + "\n"); model.train()
        return ch_acc * 0.5 + float(macro) * 0.5
    model.train(); step = 0; best = -1; t0 = time.time(); run = 0.0; n = 0
    while step < a.max_steps:
        for g in opt.param_groups: g["lr"] = a.lr * lr_mult(step)
        x, sizes, labels = q.get(); x = x.to(dev, non_blocking=True); labels = labels.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ae = model.get_audio_features(input_features=x, is_longer=torch.zeros(len(x), 1, dtype=torch.bool, device=dev))
        aen = torch.nn.functional.normalize(ae.float(), dim=-1); L = head(aen * 10.0); BL = bag_logits(L, sizes, labels)
        loss = torch.nn.functional.cross_entropy(BL, labels, weight=cw)
        if a.contrast_weight > 0:   # contrastive on the top window of each positive bag vs its class text
            tops, tl = [], []; o = 0
            for nb, y in zip(sizes, labels.tolist()):
                if y != cid[BG]: tops.append(aen[o:o + nb][L[o:o + nb, y].argmax()]); tl.append(y)
                o += nb
            if len(tops) >= 2:
                t = proc(text=[random.choice(texts[classes[y]]) for y in tl], return_tensors="pt", padding=True).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): te = model.get_text_features(**t)
                te = torch.nn.functional.normalize(te.float(), dim=-1); A = torch.stack(tops); scale = model.logit_scale_a.exp().clamp(max=100); yl = torch.tensor(tl, device=dev)
                pos = (yl[:, None] == yl[None, :]).float()
                def mp(lg): lp = lg.log_softmax(-1); return -((lp * pos).sum(-1) / pos.sum(-1)).mean()
                loss = loss + a.contrast_weight * (mp(scale * A @ te.T) + mp(scale * te @ A.T)) / 2
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1; run += loss.item(); n += 1
        if step % 50 == 0: log(f"step {step}/{a.max_steps} loss={run/n:.4f} {step/(time.time()-t0):.2f} it/s"); run = 0; n = 0
        if step % a.eval_every == 0:
            sc = evaluate(step)
            if sc > best: best = sc; model.save_pretrained(a.out / "best"); proc.save_pretrained(a.out / "best"); torch.save(head.state_dict(), a.out / "best" / "head.pt"); json.dump({"classes": classes, "texts": texts, "topk": a.topk}, open(a.out / "best" / "vocal_meta.json", "w")); log(f"saved best score={sc:.3f}")
    evaluate(step); stop.set(); log("TRAIN_DONE")
if __name__ == "__main__": main()
