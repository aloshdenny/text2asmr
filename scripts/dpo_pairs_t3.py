#!/usr/bin/env python3
"""Preference pairs for T3 (T2A v1.2): sample several takes per script window, judge them, keep best vs worst.

The judges are the ones best-of-N already trusts at render time (generate_t2a_v1.script_errors): words the
take says that the script doesn't (repeats, invented words), words it drops, unscripted laughs (x3), and
takes whose length is far off the script's (babble or a truncated ending). Training on these preferences
should make the first take as good as the best of several.

Conditioning matches rendering: a voice-bank reference (speaker embedding + its speech tokens as the prompt),
the window's text with its inline tags, the render settings (temperature 0.65, cfg 0.3, repetition 1.2).
Texts come from the training manifest; held-out creators' windows are kept apart for evaluation.

  python dpo_pairs_t3.py --voices D:\\t2a\\voice_bank_clean --out D:\\t2a\\dpo --n 1500 --takes 4
  python dpo_pairs_t3.py ... --split eval --n 200 --takes 1 --adapter <candidate>     (first-take eval)
"""
from __future__ import annotations
import argparse, difflib, glob, json, random, re, sys, time, zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_t2a_v1 as g

FILLERS = g.FILLERS
LAUGH = re.compile(r"(ha|he|hah|heh)+h?")


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def is_eval_creator(c: str, pct: float = 2.0) -> bool:                  # the split train_t3_v2 used
    return zlib.crc32((c or "").encode()) % 10000 < pct * 100


def judge(text: str, heard_text: str, n_tokens: int) -> dict:
    real = [w for w in (g._norm(t) for t in g.TAG.sub(" ", text).split()) if w and w not in FILLERS]   # "uh-uh-uh" is not a word
    heard = [g._norm(w) for w in heard_text.split() if g._norm(w)]
    laughs = 0 if re.search(r"laugh|\bha", text, re.I) else sum(bool(LAUGH.fullmatch(w)) for w in heard)
    heard = [w for w in heard if w not in FILLERS and not LAUGH.fullmatch(w)]
    ops = difflib.SequenceMatcher(a=real, b=heard, autojunk=False).get_opcodes()
    word_err = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")
    ratio = (n_tokens / 25.0) / max(1.0, g.predicted_s(text))
    length_bad = ratio < 0.5 or ratio > 1.8
    return {"word_err": word_err, "laughs": laughs, "len_ratio": round(ratio, 2),
            "score": word_err + 3 * laughs + (3 if length_bad else 0), "n_words": len(real)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="aoxo/text2asmr-t3-v2.1")
    ap.add_argument("--voices", type=Path, required=True, help="dir of (cleaned) reference wavs")
    ap.add_argument("--manifest-repo", default="aoxo/t2a-speech-v2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "eval"], default="train")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--takes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from chatterbox.tts import punc_norm, drop_invalid_tokens
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    import torch.nn.functional as F
    import librosa
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    man = [json.loads(l) for l in open(hf_hub_download(a.manifest_repo, "manifests/speech_windows_v2.jsonl", repo_type="dataset",
                                                         local_dir=str(a.out / "dl")), encoding="utf-8")]
    rng = random.Random(a.seed)
    pool = [r for r in man if (is_eval_creator(r["creator"]) == (a.split == "eval")) and 4 <= r["dur"] <= 18
            and len(r["text"].split()) >= 6]
    # tagged windows are where takes go wrong (reading tags, babbling through pauses, laughing): half of the set
    tagged = [r for r in pool if "[" in r["text"]]; plain = [r for r in pool if "[" not in r["text"]]
    rng.shuffle(tagged); rng.shuffle(plain)
    rows = (tagged[: a.n // 2] + plain[: a.n - min(len(tagged), a.n // 2)])[: a.n]
    rng.shuffle(rows)
    voices = sorted(glob.glob(str(a.voices / "*.wav")))
    log(f"{len(rows)} {a.split} windows ({sum('[' in r['text'] for r in rows)} tagged), {len(voices)} voices, {a.takes} takes each")

    model = g.load("cuda", a.adapter)
    hp = model.t3.hp
    conds = {}
    for v in voices:
        model.prepare_conditionals(v, exaggeration=0.35)
        conds[v] = (model.conds.t3.speaker_emb.clone(), model.conds.t3.cond_prompt_speech_tokens[:, -g.PROMPT_TOKENS:].clone(),
                    {k: (x.clone() if torch.is_tensor(x) else x) for k, x in model.conds.gen.items()})
    # several ASR judges, averaged: one Whisper's spelling noise ("Sand"/"San Hollow") is the same size as a real
    # 1-word difference between takes; averaged, small differences become trustworthy (weak supervision)
    judges = []
    for name in ("openai/whisper-medium.en", "openai/whisper-small.en"):
        judges.append((WhisperProcessor.from_pretrained(name),
                       WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()))

    rec_f = a.out / f"{a.split}_records.jsonl"
    done = {json.loads(l)["uid"] for l in open(rec_f, encoding="utf-8")} if rec_f.exists() else set()
    tok_f = a.out / f"{a.split}_tokens.pt"
    tokens = torch.load(tok_f) if tok_f.exists() else {}
    t0, n_new = time.time(), 0
    for i, r in enumerate(rows):
        if r["uid"] in done: continue
        v = voices[zlib.crc32(r["uid"].encode()) % len(voices)]
        spk, prompt, gen = conds[v]
        cond = T3Cond(speaker_emb=spk, cond_prompt_speech_tokens=prompt, emotion_adv=0.35 * torch.ones(1, 1, 1)).to(device="cuda")
        tt = model.tokenizer.text_to_tokens(punc_norm(r["text"])).to("cuda")
        tt = torch.cat([tt, tt], 0)
        tt = F.pad(F.pad(tt, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token)
        takes, wavs = [], []
        for j in range(a.takes):
            torch.manual_seed(a.seed * 1_000_003 + zlib.crc32(r["uid"].encode()) + 7919 * j)
            with torch.inference_mode():
                st = model.t3.inference(t3_cond=cond, text_tokens=tt, max_new_tokens=520, temperature=0.65, cfg_weight=0.3,
                                        repetition_penalty=1.2, min_p=0.05, top_p=1.0)[0]
                st = drop_invalid_tokens(st); st = st[st < 6561]
                wav, _ = model.s3gen.inference(speech_tokens=st.to("cuda"), ref_dict=gen)
            takes.append(st.cpu().to(torch.int16))
            wavs.append(librosa.resample(wav.squeeze(0).float().cpu().numpy(), orig_sr=model.sr, target_sr=16000))
        heard_by = []
        with torch.inference_mode():
            for proc, asr in judges:
                feats = proc(wavs, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
                heard_by.append(proc.batch_decode(asr.generate(feats, max_new_tokens=220), skip_special_tokens=True))
        judged = []
        for j, t in enumerate(takes):
            per = [judge(r["text"], hb[j], len(t)) for hb in heard_by]
            judged.append({"take": j, "n_tokens": len(t), "len_ratio": per[0]["len_ratio"], "n_words": per[0]["n_words"],
                           "word_err": [p["word_err"] for p in per], "laughs": [p["laughs"] for p in per],
                           "score": round(float(np.mean([p["score"] for p in per])), 2), "heard": heard_by[0][j].strip()[:300]})
        tokens[r["uid"]] = takes
        with open(rec_f, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"uid": r["uid"], "creator": r["creator"], "text": r["text"], "voice": Path(v).name,
                                 "takes": judged}) + "\n")
        n_new += 1
        if n_new % 20 == 0:
            torch.save(tokens, tok_f)
            sc = [t["score"] for l in open(rec_f, encoding="utf-8") for t in json.loads(l)["takes"]]
            log(f"  {i + 1}/{len(rows)} windows, {(time.time() - t0) / n_new:.1f} s/window | mean take score {np.mean(sc):.2f}, "
                f"clean takes {np.mean([s == 0 for s in sc]):.0%}")
    torch.save(tokens, tok_f)
    log("PAIRS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
