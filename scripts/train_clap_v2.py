#!/usr/bin/env python3
"""CLAP v2: multi-positive contrastive fine-tune on fp16 mel shards from prep_clap_v2.py."""
from __future__ import annotations
import argparse, json, math, os, random, threading, time
from collections import defaultdict
from pathlib import Path
from queue import Queue
import numpy as np

FRAMES = 1001; MELS = 64
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

class Store:
    def __init__(self, d: Path):
        self.d = d; self.mm = {}
        self.rows = [json.loads(l) for l in open(d / "index.jsonl") if l.strip()]
    def get(self, r):
        s = r["shard"]
        if s not in self.mm or r["row"] >= self.mm[s].shape[0]:
            p = self.d / f"shard_{s:04d}.f16"
            n = os.path.getsize(p) // (FRAMES * MELS * 2)
            if r["row"] >= n: raise IndexError(f"shard {s} row {r['row']} not on disk yet ({n})")
            self.mm[s] = np.memmap(p, dtype=np.float16, mode="r", shape=(n, FRAMES, MELS))
        return np.asarray(self.mm[s][r["row"]])

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, default=Path("/workspace/clap-v2"))
    ap.add_argument("--base", default="laion/clap-htsat-unfused"); ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--epochs", type=float, default=5); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--text-lr", type=float, default=1e-5); ap.add_argument("--freeze-text-frac", type=float, default=0.15)
    ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000); ap.add_argument("--push-every", type=int, default=3000)
    ap.add_argument("--push-to", default=""); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wait-for-prep", action="store_true", help="start when index has >= --min-rows and keep ingesting")
    ap.add_argument("--min-rows", type=int, default=100_000); ap.add_argument("--precision", default="bf16", choices=["bf16","fp32"]); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--overfit-n", type=int, default=0); ap.add_argument("--freeze-audio-steps", type=int, default=0, help="freeze audio_model (backbone) for the first N steps"); ap.add_argument("--total-rows", type=int, default=0, help="expected train rows once prep finishes (for step budget)")
    a = ap.parse_args()
    import torch
    from transformers import ClapModel, ClapProcessor
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    dev = "cuda"

    if a.wait_for_prep:
        while True:
            n = sum(1 for _ in open(a.data / "index.jsonl")) if (a.data / "index.jsonl").exists() else 0
            if n >= a.min_rows: break
            log(f"waiting for prep: {n}/{a.min_rows} rows"); time.sleep(60)
    store = Store(a.data)
    def split_rows():
        tr = [r for r in store.rows if r["split"] == "train"]; ev = [r for r in store.rows if r["split"] == "eval"]
        return tr, ev
    train_rows, eval_rows = split_rows()
    if a.overfit_n: train_rows = random.Random(0).sample(train_rows, a.overfit_n); eval_rows = train_rows
    classes = sorted({r["label"] for r in store.rows}); cid = {c: i for i, c in enumerate(classes)}
    log(f"rows train={len(train_rows)} eval={len(eval_rows)} classes={len(classes)}")

    proc = ClapProcessor.from_pretrained(a.base)
    model = ClapModel.from_pretrained(a.base).to(dev)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval(); [p.requires_grad_(False) for p in m.parameters()]
    text_params = [p for n, p in model.named_parameters() if n.startswith("text_model") or n.startswith("text_projection")]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith("text_model") or n.startswith("text_projection"))]
    def groups(params_named, lr):
        dec = [p for n, p in params_named if p.ndim >= 2 and "logit_scale" not in n]
        nodec = [p for n, p in params_named if not (p.ndim >= 2 and "logit_scale" not in n)]
        return [{"params": dec, "lr": lr, "weight_decay": 0.01}, {"params": nodec, "lr": lr, "weight_decay": 0.0}]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    text_named = [(n, p) for n, p in named if n.startswith("text_model") or n.startswith("text_projection")]
    other_named = [(n, p) for n, p in named if not (n.startswith("text_model") or n.startswith("text_projection"))]
    opt = torch.optim.AdamW(groups(other_named, a.lr) + groups(text_named, a.text_lr), betas=(0.9, 0.98))
    base_lrs = [a.lr, a.lr, a.text_lr, a.text_lr]
    steps_per_epoch = (a.total_rows or len(train_rows)) // a.batch
    total = a.max_steps or int(a.epochs * steps_per_epoch); freeze_until = int(a.freeze_text_frac * total)
    def lr_mult(step):
        if step < a.warmup: return step / a.warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - a.warmup) / max(1, total - a.warmup))))
    log(f"steps/epoch={steps_per_epoch} total_steps={total} text_frozen_until={freeze_until}")

    # class prompt embeddings for zero-shot eval
    class_texts = {}
    for r in store.rows:
        if r["label"] not in class_texts: class_texts[r["label"]] = list(r["text"])
    @torch.no_grad()
    def class_embs():
        model.eval(); out = []
        for c in classes:
            t = proc(text=class_texts[c], return_tensors="pt", padding=True).to(dev)
            e = torch.nn.functional.normalize(model.get_text_features(**t), dim=-1).mean(0)
            out.append(torch.nn.functional.normalize(e, dim=-1))
        model.train(); return torch.stack(out)
    @torch.no_grad()
    def evaluate(step):
        model.eval(); ce = class_embs(); correct = 0; per = defaultdict(lambda: [0, 0]); aes = []
        rows = random.Random(1).sample(eval_rows, min(6000, len(eval_rows)))
        for i in range(0, len(rows), 256):
            chunk = rows[i:i+256]
            x = torch.from_numpy(np.stack([store.get(r) for r in chunk]).astype(np.float32)).unsqueeze(1).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ae = model.get_audio_features(input_features=x, is_longer=torch.zeros(len(chunk), 1, dtype=torch.bool, device=dev))
            ae = torch.nn.functional.normalize(ae.float(), dim=-1); pred = (ae @ ce.T).argmax(-1).tolist(); aes.append(ae[:64])
            for r, p in zip(chunk, pred):
                per[r["label"]][1] += 1
                if classes[p] == r["label"]: per[r["label"]][0] += 1; correct += 1
        acc = correct / len(rows); rec = {c: round(v[0] / v[1], 3) for c, v in per.items() if v[1]}
        macro = sum(rec.values()) / len(rec)
        A = torch.cat(aes)[:512]; S = A @ A.T; acos = S[~torch.eye(len(A), dtype=torch.bool, device=S.device)].mean().item()
        log(f"EVAL step={step} top1={acc:.4f} macro_recall={macro:.4f} audio_cos={acos:.3f} scale={model.logit_scale_a.exp().item():.1f} " + " ".join(f"{c}={v}" for c, v in sorted(rec.items(), key=lambda x: -per[x[0]][1])))
        (a.out / "eval.jsonl").open("a").write(json.dumps({"step": step, "top1": acc, "macro_recall": macro, "audio_cos": acos, "recall": rec}) + "\n")
        model.train(); return acc, macro

    by_class = defaultdict(list)
    for r in train_rows: by_class[r["label"]].append(r)
    cls_list = list(by_class); cls_w = np.array([len(by_class[c]) for c in cls_list], dtype=np.float64) ** 0.7; cls_w /= cls_w.sum()
    q: Queue = Queue(maxsize=6); stop = threading.Event()
    def feeder():
        rng = random.Random(random.random())
        while not stop.is_set():
          try:
            picks = [by_class[cls_list[i]][rng.randrange(len(by_class[cls_list[i]]))] for i in np.random.choice(len(cls_list), a.batch, p=cls_w)]
            x = torch.from_numpy(np.stack([store.get(r) for r in picks]).astype(np.float32)).unsqueeze(1)
            texts = [rng.choice(r["text"]) for r in picks]; labels = torch.tensor([cid[r["label"]] for r in picks])
            t = proc(text=texts, return_tensors="pt", padding=True)
            q.put((x.pin_memory(), {k: v.pin_memory() for k, v in t.items()}, labels))
          except Exception as e:
            log(f"feeder err: {type(e).__name__}: {e}"); time.sleep(1)
    for _ in range(3): threading.Thread(target=feeder, daemon=True).start()

    def multi_pos_loss(ae, te, labels, scale):
        ae = torch.nn.functional.normalize(ae, dim=-1); te = torch.nn.functional.normalize(te, dim=-1)
        logits = scale * ae @ te.T; pos = (labels[:, None] == labels[None, :]).float()
        def one_dir(lg):
            lp = lg.log_softmax(-1); return -((lp * pos).sum(-1) / pos.sum(-1)).mean()
        return (one_dir(logits) + one_dir(logits.T)) / 2

    a.out.mkdir(parents=True, exist_ok=True)
    def save(step, tag="ckpt"):
        model.save_pretrained(a.out); proc.save_pretrained(a.out)
        (a.out / "finetune_meta.json").write_text(json.dumps({"step": step, "total": total, "base": a.base, "batch": a.batch, "classes": classes, "tag": tag}, indent=2))
        log(f"saved {tag} step={step}")
    def push(step):
        if not a.push_to: return
        try:
            from huggingface_hub import HfApi
            HfApi().upload_folder(folder_path=str(a.out), repo_id=a.push_to, repo_type="model", commit_message=f"clap v2 step {step}", ignore_patterns=["*.tmp"])
            log(f"pushed step={step} -> {a.push_to}")
        except Exception as e: log(f"push failed: {e}")

    model.train(); [p.requires_grad_(False) for p in text_params]
    audio_backbone = [p for n, p in model.named_parameters() if n.startswith("audio_model")]
    if a.freeze_audio_steps > 0:
        [p.requires_grad_(False) for p in audio_backbone]; log(f"audio backbone frozen for {a.freeze_audio_steps} steps")
    step = 0; t0 = time.time(); run = 0.0; n = 0; last_rows = len(store.rows)
    while step < total:
        if a.freeze_audio_steps and step == a.freeze_audio_steps:
            [p.requires_grad_(True) for p in audio_backbone]; log("audio backbone unfrozen")
        if step == freeze_until:
            [p.requires_grad_(True) for p in text_params]; log("text tower unfrozen")
        if a.wait_for_prep and step % 500 == 0:
            cur = sum(1 for _ in open(a.data / "index.jsonl"))
            if cur > last_rows:
                store.rows = [json.loads(l) for l in open(a.data / "index.jsonl") if l.strip()]; train_rows, eval_rows = split_rows()
                by_class.clear()
                for r in train_rows: by_class[r["label"]].append(r)
                last_rows = cur; log(f"ingested rows={cur} train={len(train_rows)}")
        for g, base in zip(opt.param_groups, base_lrs): g["lr"] = base * lr_mult(step)
        x, t, labels = q.get(); x = x.to(dev, non_blocking=True); labels = labels.to(dev); t = {k: v.to(dev, non_blocking=True) for k, v in t.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.precision == "bf16")):
            ae = model.get_audio_features(input_features=x, is_longer=torch.zeros(len(x), 1, dtype=torch.bool, device=dev))
            te = model.get_text_features(**t)
        scale = model.logit_scale_a.exp().clamp(max=100) if hasattr(model, "logit_scale_a") else model.logit_scale.exp().clamp(max=100)
        loss = multi_pos_loss(ae.float(), te.float(), labels, scale)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
        run += loss.item(); n += 1
        if step % 50 == 0:
            el = time.time() - t0; ips = step / el
            log(f"step {step}/{total} loss={run/n:.4f} {ips:.2f} it/s ETA {(total-step)/ips/60:.0f} min q={q.qsize()}"); run = 0.0; n = 0
        if step % a.eval_every == 0: evaluate(step)
        if step % a.save_every == 0: save(step)
        if step % a.push_every == 0: push(step)
    evaluate(step); save(step, "final"); push(step); stop.set()
    log("TRAIN_DONE"); return 0

if __name__ == "__main__":
    raise SystemExit(main())
