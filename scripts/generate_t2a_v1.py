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


def quietest(wav: np.ndarray, sr: int, t0: float, t1: float, win: float = 0.04) -> float | None:
    """Centre (s) of the quietest `win` window inside [t0, t1]: where a cut or a pause can go without touching
    a word. Whisper's word times are often 100-200 ms off, so edits are placed by the audio, not by them."""
    a, b, w = int(max(0.0, t0) * sr), int(t1 * sr), int(win * sr)
    if b - a < w: return None
    hop = max(1, w // 2)
    e = [(float(np.mean(wav[i:i + w] ** 2)), i + w // 2) for i in range(a, b - w, hop)]
    return min(e)[1] / sr if e else None


def room_noise(wav: np.ndarray, sr: int, dur: float, level: float | None = None) -> np.ndarray:
    """Room tone synthesised from the chunk's own quietest frames: random-offset, Hann-windowed overlap-add, so
    nothing repeats (a looped slice is a 10 Hz buzz) and the spectrum is the room's, level-matched to `level`."""
    n = int(dur * sr); w = int(0.05 * sr); hop = w // 2
    frames = [wav[i:i + w] for i in range(0, max(1, len(wav) - w), hop)]
    if n <= 0 or not frames: return np.zeros(max(0, n), np.float32)
    e = np.array([np.mean(f ** 2) for f in frames])
    pool = [f for f, x in zip(frames, e) if x <= np.percentile(e, 15)] or frames[:1]
    rng = np.random.default_rng(n + len(wav)); win = np.hanning(w).astype(np.float32)
    out = np.zeros(n + w, np.float32)
    for k in range(0, n, hop):
        out[k:k + w] += pool[rng.integers(len(pool))][:w] * win
    out = out[:n]
    target = level if level is not None else float(np.sqrt(np.mean(np.concatenate(pool) ** 2)))
    return (out * (target / (float(np.sqrt(np.mean(out ** 2))) + 1e-12))).astype(np.float32)


def xfade(a: np.ndarray, b: np.ndarray, sr: int, ms: float = 30) -> np.ndarray:
    """Equal-power crossfade join: no click, no level dip at the seam."""
    n = min(int(ms / 1000 * sr), len(a) // 2, len(b) // 2)
    if n <= 0: return np.concatenate([a, b])
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    return np.concatenate([a[:-n], a[-n:] * np.cos(t) + b[:n] * np.sin(t), b[n:]]).astype(np.float32)


def local_level(wav: np.ndarray, sr: int, t: float, span: float = 0.05) -> float:
    seg = wav[max(0, int((t - span) * sr)): int((t + span) * sr)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def align_trim(wav: np.ndarray, sr: int, text: str, lookahead: list[str], event_pause: float = 0.5) -> tuple[np.ndarray, str]:
    """Trim the chunk to its own words and enforce its pauses, with every edit in silence.

    * look-ahead: the chunk was generated with the next chunk's first words appended, so its real last word is
      spoken fully; the end cut goes at the quietest point between that word and the look-ahead
    * the start cut goes at the quietest point before the first word -- or nowhere, if the chunk opens with a tag
      (trimming to the first word cut the opening breath away)
    * [pause Ns] is stretched to at least N s; a vocal event gets `event_pause` s of quiet after it -- inserted at
      the quietest point in the latter part of the gap, i.e. after the event has decayed naturally, as room tone
      built from the chunk's own background at the level around it"""
    global _ASR
    import difflib, librosa
    from faster_whisper import WhisperModel
    if _ASR is None: _ASR = WhisperModel("small.en", device="cpu", compute_type="int8")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    words = [w for seg in _ASR.transcribe(w16, language="en", word_timestamps=True, vad_filter=False)[0] for w in (seg.words or [])]
    if not words: return wav, "no words heard"
    real = [_norm(t) for t in TAG.sub(" ", text).split() if _norm(t)]
    la = [_norm(t) for t in lookahead if _norm(t)]
    sm = difflib.SequenceMatcher(a=real + la, b=[_norm(w.word) for w in words], autojunk=False)
    s2h = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size): s2h[blk.a + k] = blk.b + k
    jw = max((s2h[i] for i in range(len(real)) if i in s2h), default=None)
    jl = min((s2h[i] for i in range(len(real), len(real) + len(la)) if i in s2h), default=None)
    if jw is None: return wav, "no script words matched"
    dropped = len(real) - 1 - max(i for i in range(len(real)) if i in s2h)
    dur = len(wav) / sr

    tags, k, pos = [], 0, 0
    for m in TAG.finditer(text):
        k += len([t for t in text[pos:m.start()].split() if _norm(t)])
        tags.append((m.group(1), float(m.group(2)) if m.group(2) else None, k - 1, k)); pos = m.end()
    if tags and tags[0][2] == -1:
        s0 = 0.0
    else:
        w0 = words[0].start
        s0 = quietest(wav, sr, w0 - 0.45, w0 - 0.08) or max(0.0, w0 - 0.25)
    end_w = words[jw].end
    hi = (words[jl].start - 0.05) if (jl is not None and jl > jw) else min(dur, end_w + 0.5)
    s1 = min(dur, quietest(wav, sr, end_w + 0.05, hi) or end_w + 0.25)

    inserts = []
    for name, secs, prev_k, next_k in tags:
        if next_k not in s2h or next_k >= len(real): continue
        t_next = words[s2h[next_k]].start
        t_prev = words[s2h[prev_k]].end if prev_k >= 0 and prev_k in s2h else s0
        if name == "pause":
            need, lo = (secs or 0.0) - (t_next - t_prev), t_prev + 0.08
        else:
            need, lo = event_pause, (t_prev + t_next) / 2        # after the event has decayed
        at = quietest(wav, sr, lo, t_next - 0.12)
        if need > 0.08 and at is not None and s0 < at < s1: inserts.append((at, need))

    out, last = None, s0
    for at, need in sorted(inserts):
        seg = wav[int(last * sr): int(at * sr)]
        gap = room_noise(wav, sr, need, local_level(wav, sr, at))
        out = seg if out is None else xfade(out, seg, sr)
        out = xfade(out, gap, sr); last = at
    seg = wav[int(last * sr): int(s1 * sr)]
    out = seg if out is None else xfade(out, seg, sr)
    added = sum(n for _, n in inserts)
    note = "ok" if dropped <= 0 else f"model dropped the last {dropped} word(s)"
    return out.astype(np.float32), note + (f", +{added:.1f}s enforced pauses" if added else "")


BREATH_CUES = [(re.compile(r"\b(breathe in|breathing in|inhale|take a (deep |slow )?breath)\b", re.I), 4.0),
               (re.compile(r"\bhold( it| that| your breath)?\b", re.I), 3.0),
               (re.compile(r"\b(breathe out|breathing out|exhale|let it (all )?(out|go))\b", re.I), 5.0)]


def add_breath_cues(script: str) -> str:
    """Give the listener time to actually do what a breathing cue asks: ~4 s to inhale, ~3 s to hold, ~5 s to
    release, after the clause that asks for it -- unless the script already put a pause there."""
    parts = re.split(r"(?<=[,.!?…])\s+", " ".join(script.split()))
    for i, c in enumerate(parts):
        if c.rstrip().endswith("]"): continue
        secs = next((sec for rx, sec in BREATH_CUES if rx.search(TAG.sub(" ", c))), None)
        if secs is None: continue
        nxt = parts[i + 1] if i + 1 < len(parts) else ""
        m = re.match(r"\[pause ([\d.]+)s\]", nxt)
        if m:                                             # the script already pauses here: at least the cue length
            if float(m.group(1)) < secs: parts[i + 1] = f"[pause {secs:.1f}s]" + nxt[m.end():]
        else:
            parts[i] = f"{c} [pause {secs:.1f}s]"
    return " ".join(parts)


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
    ap.add_argument("--temperature", type=float, default=0.65, help="lower = fewer garbled/odd words")
    ap.add_argument("--cfg", type=float, default=0.3, help="lower = slower, more deliberate pacing")
    ap.add_argument("--exaggeration", type=float, default=0.35, help="lower = gentler, calmer delivery")
    ap.add_argument("--event-pause", type=float, default=0.5, help="quiet seconds after each vocal event tag")
    ap.add_argument("--no-breath-cues", action="store_true", help="don't add inhale/hold/exhale pauses after breathing cues")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-align-trim", action="store_true", help="skip word-aligned trimming of chunk edges")
    ap.add_argument("--gap-s", type=float, default=0.45, help="silence between chunks (words never touch a seam)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    import torch.nn.functional as F
    from chatterbox.tts import punc_norm, drop_invalid_tokens
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    script = Path(a.script).read_text() if Path(a.script).exists() else a.script
    if not a.no_breath_cues: script = add_breath_cues(script)
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
            wav, note = align_trim(wav, sr, text, lookahead, a.event_pause)
        log(f"chunk {i + 1}/{len(chunks)}: {raw_s:.1f} s -> {len(wav) / sr:.1f} s ({note}; "
            f"predicted {predicted_s(text):.1f} s) in {time.time() - t0:.0f} s | {text[:70]}")
        # join through the previous chunk's own room tone at its closing level -- never digital silence
        if not out: out = [wav]
        else:
            prev = out[-1]
            gap = room_noise(prev, sr, a.gap_s, local_level(prev, sr, len(prev) / sr - 0.06))
            out[-1] = xfade(xfade(prev, gap, sr), wav, sr)
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
