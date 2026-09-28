#!/usr/bin/env python3
"""T2A v1 speech model: Chatterbox T3 LoRA on the precomputed v2 windows (prep_t3_v2.py).

What is new versus the v1 baseline (train_speech.py):
  * data: 2,000 h of packed 2-20 s windows with inline [pause Ns] / [breathing] / [oral sounds] / [moaning]
    tags, from ~6,000 creators, instead of isolated phrases
  * continuation: the 4 s of the same recording before each window goes in as the speech prompt
    (cond_prompt_speech_tokens -> perceiver). At inference a long script is generated as chained <= 20 s
    windows, each prompted with the tail of the previous one -- the same thing it trained on
  * length: speech cap 510 tokens (20 s + start/stop) instead of 400; text <= 128 tokens
  * batches are bucketed by length (token budget, not a fixed count), so 2 s and 20 s windows do not share
    padding
  * eval loss on held-out creators, never on held-out windows of seen creators

  python train_t3_v2.py --data D:\\t2a\\t3v2 --out D:\\t2a\\t3v2_ckpt --push aoxo/text2asmr-t3-v2
"""
from __future__ import annotations
import argparse, glob, json, math, random, sys, time, zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text2asmr.models.chatterbox_ft import ChatterboxFinetuner, load_backbone

PROMPT_TOKENS = 96          # last ~3.8 s of the previous audio; the perceiver resamples it to a fixed size
MAX_SPEECH = 510            # 20 s @ 25 Hz + start/stop
MAX_TEXT = 128


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def is_eval_creator(c: str, pct: float) -> bool:
    return zlib.crc32((c or "").encode()) % 10000 < pct * 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--push", default="")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--token-budget", type=int, default=7000, help="speech tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--eval-pct", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N optimizer steps")
    ap.add_argument("--max-shards", type=int, default=0, help="smoke test: load only N shards")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    backbone = load_backbone("cuda")
    tok = backbone.tokenizer; hp = backbone.t3.hp
    from chatterbox.tts import punc_norm
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    # ---- load every shard (a few GB of int16 tokens; the box has 62 GB) ----
    rows, dropped = [], {"no prompt": 0, "too long": 0, "text too long": 0}
    for f in sorted(glob.glob(str(a.data / "shard_*.pt")))[: a.max_shards or None]:
        for r in torch.load(f, weights_only=False):
            if r["prompt"] is None or len(r["prompt"]) < PROMPT_TOKENS: dropped["no prompt"] += 1; continue
            if len(r["speech"]) + 2 > MAX_SPEECH: dropped["too long"] += 1; continue
            t = torch.as_tensor(tok.text_to_tokens(punc_norm(r["text"])).squeeze(0), dtype=torch.long)
            if len(t) + 2 > MAX_TEXT: dropped["text too long"] += 1; continue
            rows.append({"text": t, "speech": r["speech"].long(), "prompt": r["prompt"][-PROMPT_TOKENS:].long(),
                         "spk": r["spk"].float(), "creator": r.get("creator") or "", "dur": r["dur"]})
    ev = [r for r in rows if is_eval_creator(r["creator"], a.eval_pct)]
    tr = [r for r in rows if not is_eval_creator(r["creator"], a.eval_pct)]
    log(f"windows: train {len(tr)} ({sum(r['dur'] for r in tr) / 3600:.0f} h), eval {len(ev)} "
        f"({len({r['creator'] for r in ev})} held-out creators); dropped {dropped}")

    def batches(rs, shuffle=True):
        """Length-bucketed batches under a speech-token budget."""
        rs = sorted(rs, key=lambda r: len(r["speech"]))
        out, cur, width = [], [], 0
        for r in rs:
            w = max(width, len(r["speech"]) + 2)
            if cur and w * (len(cur) + 1) > a.token_budget:
                out.append(cur); cur, w = [], len(r["speech"]) + 2
            cur.append(r); width = w
        if cur: out.append(cur)
        if shuffle: random.shuffle(out)
        return out

    def collate(rs):
        S, E = hp.start_text_token, hp.stop_text_token
        texts = [torch.cat([torch.tensor([S]), r["text"], torch.tensor([E])]) for r in rs]
        speech = [torch.cat([torch.tensor([hp.start_speech_token]), r["speech"], torch.tensor([hp.stop_speech_token])]) for r in rs]
        tl = torch.tensor([len(t) for t in texts]); sl = torch.tensor([len(s) for s in speech])
        tp = torch.zeros(len(rs), int(tl.max()), dtype=torch.long); sp = torch.zeros(len(rs), int(sl.max()), dtype=torch.long)
        for i, (t, s) in enumerate(zip(texts, speech)): tp[i, :len(t)] = t; sp[i, :len(s)] = s
        return {"text_tokens": tp, "text_token_lens": tl, "speech_tokens": sp, "speech_token_lens": sl,
                "speaker_emb": torch.stack([r["spk"] for r in rs]), "prompt": torch.stack([r["prompt"] for r in rs])}

    ft = ChatterboxFinetuner(backbone, lora_r=a.lora_r, lora_alpha=2 * a.lora_r, lr=a.lr, warmup=300)

    # The base finetuner builds T3Cond without a prompt; v1 conditions on the continuation prompt.
    def forward_loss(batch):
        from text2asmr.models.chatterbox_ft import _causal_lm_losses
        dev = backbone.device
        cond = T3Cond(speaker_emb=batch["speaker_emb"].to(dev), cond_prompt_speech_tokens=batch["prompt"].to(dev),
                      emotion_adv=0.5 * torch.ones(len(batch["speaker_emb"]), 1, 1, device=dev))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lt, ls = _causal_lm_losses(ft.t3, cond=cond, text_tokens=batch["text_tokens"].to(dev),
                                       text_lens=batch["text_token_lens"].to(dev), speech_tokens=batch["speech_tokens"].to(dev),
                                       speech_lens=batch["speech_token_lens"].to(dev))
        return ft.speech_loss_weight * ls + ft.text_loss_weight * lt
    ft._forward_loss = forward_loss

    step = 0
    last = a.out / "last"
    if (last / "trainer_state.pt").exists(): step = ft.resume(last)
    ev_batches = batches(ev, shuffle=False)[:200]
    total = int(len(batches(tr)) * a.epochs / a.accum)
    if a.max_steps: total = min(total, a.max_steps)
    log(f"optimizer steps: {total} (resuming at {step}); lora r={a.lora_r}")

    def evaluate():
        ft.t3.eval(); ls = [ft.loss_only(collate(b)) for b in ev_batches]; ft.t3.train()
        ls = [x for x in ls if not math.isnan(x)]
        return sum(ls) / max(1, len(ls))

    best = float("inf"); t0 = time.time(); micro = 0; run = []
    ft.t3.train()
    while step < total:
        for b in batches(tr):
            if step >= total: break
            micro += 1
            loss = ft.step(collate(b), accumulate=(micro % a.accum != 0))
            if not math.isnan(loss): run.append(loss)
            if micro % a.accum: continue
            step += 1
            if step % 50 == 0:
                log(f"step {step}/{total} loss {sum(run) / max(1, len(run)):.4f} ({(time.time() - t0) / 60:.0f} min, "
                    f"oom skips {ft.oom_skips})"); run = []
            if step % a.eval_every == 0:
                el = evaluate(); log(f"EVAL step {step} held-out-creator loss {el:.4f}")
                if el < best: best = el; ft.save(a.out / "best", step)
            if step % a.save_every == 0: ft.save(last, step)
    ft.save(last, step)
    el = evaluate(); log(f"EVAL final held-out-creator loss {el:.4f} (best {min(best, el):.4f})")
    if el < best: ft.save(a.out / "best", step)
    if a.push:
        from huggingface_hub import HfApi
        api = HfApi(); api.create_repo(a.push, exist_ok=True)
        api.upload_folder(repo_id=a.push, folder_path=str(a.out / "best"), path_in_repo="adapter",
                          commit_message=f"T2A v1 speech adapter: held-out-creator loss {min(best, el):.4f}")
        log(f"pushed -> {a.push}")
    log("TRAIN_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
