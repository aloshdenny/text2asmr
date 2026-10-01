#!/usr/bin/env python3
"""T2A v1.2: DPO on the T3 speech adapter, from preference pairs made by dpo_pairs_t3.py.

Each pair is two takes of the same script window from v1.1 with the same voice: the take the averaged judges
scored best (fewest wrong / repeated / dropped words, no unscripted laugh, sane length) is "chosen", a clearly
worse one "rejected". The policy starts at v1.1's adapter; the reference is v1.1 itself, whose log-probs are
computed once before training (same weights, eval mode), so no second model copy sits in memory.

loss = -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))  +  alpha * NLL(chosen) / len(chosen)
The NLL term keeps the chosen takes likely (plain DPO can push both down).

  python train_t3_dpo.py --pairs D:\\t2a\\dpo --voices D:\\t2a\\voice_bank_clean --out D:\\t2a\\t3dpo
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_t2a_v1 as g


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True, help="dir with train_records.jsonl + train_tokens.pt")
    ap.add_argument("--voices", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapter", default="aoxo/text2asmr-t3-v2.1")
    ap.add_argument("--min-gap", type=float, default=1.5, help="averaged judge-score gap for a pair")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--pairs-per-step", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--push", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from chatterbox.tts import ChatterboxTTS, punc_norm
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    recs = [json.loads(l) for l in open(a.pairs / "train_records.jsonl", encoding="utf-8-sig") if l.strip()]
    toks = torch.load(a.pairs / "train_tokens.pt")
    pairs = []
    for r in recs:
        if r["uid"] not in toks: continue
        ts = sorted(r["takes"], key=lambda t: t["score"]); best = ts[0]
        for worse in ts[1:]:
            if worse["score"] - best["score"] >= a.min_gap:
                pairs.append({"uid": r["uid"], "text": r["text"], "voice": r["voice"], "chosen": toks[r["uid"]][best["take"]],
                              "rejected": toks[r["uid"]][worse["take"]], "gap": worse["score"] - best["score"]})
    random.Random(0).shuffle(pairs)
    uids = sorted({p["uid"] for p in pairs}); ho = set(uids[: int(len(uids) * a.holdout)])
    tr = [p for p in pairs if p["uid"] not in ho]; ev = [p for p in pairs if p["uid"] in ho]
    log(f"{len(recs)} windows -> {len(pairs)} pairs (gap >= {a.min_gap}); train {len(tr)}, held-out {len(ev)}")

    model = ChatterboxTTS.from_pretrained("cuda"); hp = model.t3.hp
    conds = {}
    for v in sorted({p["voice"] for p in pairs}):
        model.prepare_conditionals(str(a.voices / v), exaggeration=0.35)
        conds[v] = (model.conds.t3.speaker_emb.detach().clone(),
                    model.conds.t3.cond_prompt_speech_tokens[:, -g.PROMPT_TOKENS:].detach().clone())
    path = Path(snapshot_download(a.adapter, allow_patterns=["adapter/*"])) / "adapter"
    model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, str(path), is_trainable=True)
    t3 = model.t3
    params = [p for p in t3.parameters() if p.requires_grad]
    log(f"trainable LoRA params: {sum(p.numel() for p in params) / 1e6:.1f} M")

    def seq_logps(batch: list[dict], which: str) -> torch.Tensor:
        """Summed log-prob of each take's speech tokens under the current T3, conditioned as at generation."""
        S, E = hp.start_text_token, hp.stop_text_token
        texts = [torch.cat([torch.tensor([S]), model.tokenizer.text_to_tokens(punc_norm(p["text"])).squeeze(0).cpu(), torch.tensor([E])])
                 for p in batch]
        speech = [torch.cat([torch.tensor([hp.start_speech_token]), p[which].long(), torch.tensor([hp.stop_speech_token])]) for p in batch]
        tl = torch.tensor([len(t) for t in texts]); sl = torch.tensor([len(s) for s in speech])
        tp = torch.zeros(len(batch), int(tl.max()), dtype=torch.long); sp = torch.zeros(len(batch), int(sl.max()), dtype=torch.long)
        for i, (t, s) in enumerate(zip(texts, speech)): tp[i, :len(t)] = t; sp[i, :len(s)] = s
        cond = T3Cond(speaker_emb=torch.cat([conds[p["voice"]][0] for p in batch]),
                      cond_prompt_speech_tokens=torch.cat([conds[p["voice"]][1] for p in batch]),
                      emotion_adv=0.35 * torch.ones(len(batch), 1, 1)).to(device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = t3.forward(t3_cond=cond, text_tokens=tp.cuda(), text_token_lens=tl.cuda(), speech_tokens=sp.cuda(),
                             speech_token_lens=sl.cuda(), training=True)
        logits = out.speech_logits[:, :-1].float(); tgt = sp[:, 1:].cuda()
        lp = torch.log_softmax(logits, -1).gather(-1, tgt[..., None]).squeeze(-1)
        mask = (torch.arange(tgt.size(1), device="cuda")[None] + 1 < sl.cuda()[:, None]).float()
        return (lp * mask).sum(1), mask.sum(1)

    # reference log-probs: v1.1 itself, computed once
    t3.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), 8):
            b = pairs[i:i + 8]
            rc, _ = seq_logps(b, "chosen"); rr, _ = seq_logps(b, "rejected")
            for p, c, r in zip(b, rc.tolist(), rr.tolist()): p["ref_c"], p["ref_r"] = c, r
    log("reference log-probs done")

    def evaluate() -> tuple[float, float]:
        t3.eval(); acc, marg = [], []
        with torch.no_grad():
            for i in range(0, len(ev), 8):
                b = ev[i:i + 8]
                pc, _ = seq_logps(b, "chosen"); pr, _ = seq_logps(b, "rejected")
                m = (pc - torch.tensor([p["ref_c"] for p in b], device="cuda")) - (pr - torch.tensor([p["ref_r"] for p in b], device="cuda"))
                acc += (m > 0).float().tolist(); marg += m.tolist()
        t3.train()
        return sum(acc) / max(1, len(acc)), sum(marg) / max(1, len(marg))

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(tr) / a.pairs_per_step / a.accum) * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / max(1, total)))
    acc0, m0 = evaluate(); log(f"before: held-out reward accuracy {acc0:.0%}, margin {m0:+.3f}  (v1.1 = reference, so ~0)")
    best, step, t0 = -1e9, 0, time.time()
    t3.train()
    for ep in range(a.epochs):
        random.shuffle(tr); run = []
        for i in range(0, len(tr), a.pairs_per_step):
            b = tr[i:i + a.pairs_per_step]
            pc, nc = seq_logps(b, "chosen"); pr, _ = seq_logps(b, "rejected")
            rc = torch.tensor([p["ref_c"] for p in b], device="cuda"); rr = torch.tensor([p["ref_r"] for p in b], device="cuda")
            logits = a.beta * ((pc - rc) - (pr - rr))
            loss = (-F.logsigmoid(logits)).mean() + a.alpha * (-(pc / nc)).mean()
            (loss / a.accum).backward(); run.append(loss.item())
            if (i // a.pairs_per_step + 1) % a.accum: continue
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 20 == 0:
                log(f"epoch {ep + 1} step {step}/{total} loss {sum(run) / len(run):.4f} ({(time.time() - t0) / 60:.0f} min)"); run = []
        acc, m = evaluate(); log(f"EVAL epoch {ep + 1}: held-out reward accuracy {acc:.0%}, margin {m:+.3f}")
        if m > best:
            best = m; t3.tfmr.save_pretrained(str(a.out / "best")); log(f"saved -> {a.out / 'best'}")
    if a.push:
        from huggingface_hub import HfApi
        api = HfApi(); api.create_repo(a.push, exist_ok=True)
        api.upload_folder(repo_id=a.push, folder_path=str(a.out / "best"), path_in_repo="adapter",
                          allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
                          commit_message=f"T2A v1.2 DPO adapter: held-out reward margin {best:+.3f}")
        log(f"pushed -> {a.push}")
    log("DPO_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
