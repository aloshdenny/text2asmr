#!/usr/bin/env python3
"""T2A v1 inference: a script of any length -> one ASMR audio file, as chained <= 20 s windows.

    python generate_t2a_v1.py --ref voice.wav --script script.txt --out out.wav [--adapter aoxo/text2asmr-t3-v2]

The script is plain text with the same inline tags the model trained on:
    Hey... [pause 1.5s] you look tired. [breathing] Come here, [oral sounds] let me take care of you.

How it stays inside what the model has seen (docs/GENERATOR_V2.md):
  * the planner cuts the script at sentence ends into chunks predicted to run <= --max-chunk-s (18 s): words /
    2.6 w/s + pause seconds + ~1.5 s per vocal event. No chunk asks for more than a 20 s training window.
  * chunk 1 is prompted with the reference clip's speech; every later chunk with the last 96 speech tokens of
    the chunk before -- exactly the continuation the model was trained on -- so long pieces do not drift.
  * the voice (timbre) always comes from the reference clip via S3Gen's reference embedding.
  * output carries Chatterbox's Perth watermark.
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path

import numpy as np
import torch

PROMPT_TOKENS = 96
WPS = 2.6
EVENT_S = 1.5
TAG = re.compile(r"\[([a-z ]+?)(?: ([\d.]+)s)?\]")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def predicted_s(text: str) -> float:
    s = 0.0
    for m in TAG.finditer(text):
        s += float(m.group(2)) if m.group(1) == "pause" and m.group(2) else EVENT_S
    words = len(TAG.sub(" ", text).split())
    return s + words / WPS


def plan(script: str, max_s: float) -> list[str]:
    """Sentence-level chunks, each predicted to fit one window. A sentence that alone exceeds the limit is split
    at commas, then at word boundaries, so no chunk is ever asked for more than the model trained on."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", " ".join(script.split())) if s.strip()]
    pieces = []
    for s in sents:
        if predicted_s(s) <= max_s: pieces.append(s); continue
        for part in re.split(r"(?<=,)\s+", s):
            words, cur = part.split(), []
            for w in words:
                if cur and predicted_s(" ".join(cur + [w])) > max_s: pieces.append(" ".join(cur)); cur = []
                cur.append(w)
            if cur: pieces.append(" ".join(cur))
    chunks, cur = [], ""
    for p in pieces:
        cand = f"{cur} {p}".strip()
        if cur and predicted_s(cand) > max_s: chunks.append(cur); cur = p
        else: cur = cand
    if cur: chunks.append(cur)
    return chunks


_ASR = None
_norm = lambda w: re.sub(r"[^a-z']", "", w.lower())


def align_trim(wav: np.ndarray, sr: int, text: str, lookahead: list[str]) -> tuple[np.ndarray, str]:
    """Cut the chunk to [first word - 120 ms, the pause between its last real word and the look-ahead].

    An autoregressive TTS under-pronounces the final word of its input: it stops early ("brea-") or the word
    mutates. So every chunk is generated with the next chunk's first words appended (look-ahead): the real last
    word is never the sequence end and gets spoken fully. Whisper word timings, aligned to the expected words,
    find the pause after the real last word; the cut goes midway through that pause and the look-ahead audio is
    thrown away. The next chunk says those words for real."""
    global _ASR
    import difflib, librosa
    from faster_whisper import WhisperModel
    if _ASR is None: _ASR = WhisperModel("small.en", device="cpu", compute_type="int8")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    words = [w for seg in _ASR.transcribe(w16, language="en", word_timestamps=True, vad_filter=False)[0] for w in (seg.words or [])]
    if not words: return wav, "no words heard"
    real = [_norm(t) for t in TAG.sub(" ", text).split() if _norm(t)]
    la = [_norm(t) for t in lookahead if _norm(t)]
    heard = [_norm(w.word) for w in words]
    sm = difflib.SequenceMatcher(a=real + la, b=heard, autojunk=False)
    s2h = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size): s2h[blk.a + k] = blk.b + k
    jw = max((s2h[i] for i in range(len(real)) if i in s2h), default=None)          # last real word heard
    jl = min((s2h[i] for i in range(len(real), len(real) + len(la)) if i in s2h), default=None)  # first look-ahead
    if jw is None: return wav, "no script words matched"
    end_w = words[jw].end
    if jl is not None and jl > jw:
        cut = max(end_w + 0.05, (end_w + words[jl].start) / 2)                         # middle of the pause
    else:
        cut = end_w + 0.3                                                              # look-ahead not spoken
    dropped = len(real) - 1 - max(i for i in range(len(real)) if i in s2h)
    s0 = max(0.0, words[0].start - 0.12); s1 = min(len(wav) / sr, cut)
    out = wav[int(s0 * sr): int(s1 * sr)].copy()
    f = min(int(0.03 * sr), len(out) // 2)
    if f: out[:f] *= np.linspace(0, 1, f); out[-f:] *= np.linspace(1, 0, f)
    return out, ("ok" if dropped <= 0 else f"model dropped the last {dropped} word(s)")


def load(device: str, adapter: str | None):
    from chatterbox.tts import ChatterboxTTS
    if device == "mps":                                  # chatterbox checkpoints are saved on cuda; map them
        _load = torch.load
        torch.load = lambda *a, **k: _load(*a, **{**k, "map_location": torch.device("cpu")})
    model = ChatterboxTTS.from_pretrained(device)
    if adapter:
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        path = Path(adapter)
        if not path.exists():
            path = Path(snapshot_download(adapter, allow_patterns=["adapter/*"])) / "adapter"
        model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, str(path)).merge_and_unload()
        log(f"adapter merged from {adapter}")
    model.t3.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference voice clip (6-12 s of clean speech)")
    ap.add_argument("--script", required=True, help="text file, or the script itself")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="aoxo/text2asmr-t3-v2", help="HF repo or local dir; '' for the base model")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-chunk-s", type=float, default=18.0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--cfg", type=float, default=0.3, help="lower = slower, more deliberate pacing")
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-align-trim", action="store_true", help="skip word-aligned trimming of chunk edges")
    ap.add_argument("--gap-s", type=float, default=0.45, help="silence between chunks (words never touch a seam)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    import torch.nn.functional as F
    from chatterbox.tts import punc_norm, drop_invalid_tokens
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    script = Path(a.script).read_text() if Path(a.script).exists() else a.script
    chunks = plan(script, a.max_chunk_s)
    log(f"{len(chunks)} chunks, predicted {sum(predicted_s(c) for c in chunks):.0f} s")
    model = load(a.device, a.adapter or None)
    model.prepare_conditionals(a.ref, exaggeration=a.exaggeration)
    base = model.conds.t3
    prompt = base.cond_prompt_speech_tokens[:, -PROMPT_TOKENS:]          # same prompt length as training
    hp = model.t3.hp
    out, sr = [], model.sr
    TAIL = ["Okay", "then."]                                  # look-ahead for the final chunk; always cut off
    for i, text in enumerate(chunks):
        lookahead = (TAG.sub(" ", chunks[i + 1]).split()[:3] if i + 1 < len(chunks) else TAIL)
        gen_text = text if a.no_align_trim else f"{text} {' '.join(lookahead)}"
        cond = T3Cond(speaker_emb=base.speaker_emb, cond_prompt_speech_tokens=prompt,
                      emotion_adv=a.exaggeration * torch.ones(1, 1, 1)).to(device=model.device)
        tt = model.tokenizer.text_to_tokens(punc_norm(gen_text)).to(model.device)
        if a.cfg > 0: tt = torch.cat([tt, tt], dim=0)
        tt = F.pad(F.pad(tt, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token)
        t0 = time.time()
        with torch.inference_mode():
            st = model.t3.inference(t3_cond=cond, text_tokens=tt, max_new_tokens=520, temperature=a.temperature,
                                    cfg_weight=a.cfg, repetition_penalty=1.2, min_p=0.05, top_p=1.0)[0]
            st = drop_invalid_tokens(st); st = st[st < 6561].to(model.device)
            wav, _ = model.s3gen.inference(speech_tokens=st, ref_dict=model.conds.gen)
        wav = wav.squeeze(0).detach().cpu().numpy()
        raw_s = len(wav) / sr
        note = "untrimmed"
        if not a.no_align_trim:
            wav, note = align_trim(wav, sr, text, lookahead)
        log(f"chunk {i + 1}/{len(chunks)}: {raw_s:.1f} s -> {len(wav) / sr:.1f} s ({note}; "
            f"predicted {predicted_s(text):.1f} s) in {time.time() - t0:.0f} s | {text[:70]}")
        out += [wav, np.zeros(int(a.gap_s * sr), dtype=np.float32)]
        # continuation prompt from the *trimmed* tail, so a garbled last word never seeds the next chunk
        import librosa
        tail16 = librosa.resample(wav[-int(6 * sr):], orig_sr=sr, target_sr=16000)
        with torch.inference_mode():
            pt, pl = model.s3gen.tokenizer.forward([tail16])
        if int(pl[0]) >= PROMPT_TOKENS:
            prompt = pt[:, :int(pl[0])][:, -PROMPT_TOKENS:].long().to(model.device)
    audio = np.concatenate(out)
    audio = model.watermarker.apply_watermark(audio, sample_rate=sr)
    import soundfile as sf
    sf.write(a.out, audio, sr)
    log(f"wrote {a.out}: {len(audio) / sr:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
