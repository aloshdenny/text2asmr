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
MOAN = re.compile(r"\[moaning\]|\b(m{2,}|mm+h?m*|o+h+|a+h+)\b", re.I)


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
    """Room tone with the room's spectrum and no time structure: random-phase noise shaped to the mean spectrum of
    the chunk's quietest frames, level-matched to `level`. (A looped slice buzzed at 10 Hz; overlap-adding random
    grains fluttered at the grain rate, 40 Hz -- stationary noise has nothing to repeat or flutter.)"""
    import librosa
    n = int(dur * sr)
    if n <= 0: return np.zeros(max(0, n), np.float32)
    S = np.abs(librosa.stft(wav, n_fft=1024, hop_length=256)) ** 2
    e = S.sum(0); quiet = e <= np.percentile(e, 15)
    prof = np.sqrt(S[:, quiet].mean(1))
    rng = np.random.default_rng(n + len(wav))
    X = prof[:, None] * np.exp(2j * np.pi * rng.random((len(prof), n // 256 + 8)))
    out = librosa.istft(X, hop_length=256, n_fft=1024, length=n + 2048)[1024:1024 + n]
    if level is None:
        idx = librosa.frames_to_samples(np.where(quiet)[0], hop_length=256)
        level = float(np.sqrt(np.mean(wav[np.minimum(idx, len(wav) - 1)] ** 2)))
    return (out * (level / (float(np.sqrt(np.mean(out ** 2))) + 1e-12))).astype(np.float32)


def tempo(wav: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Change tempo without changing pitch (ffmpeg atempo). Applied to the finished piece, pauses included:
    per chunk, slowed audio would seed the next chunk's continuation prompt and the slowdown would compound."""
    import subprocess
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                        "-filter:a", f"atempo={factor}", "-f", "f32le", "pipe:1"],
                       input=wav.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).copy()


def gate(audio: np.ndarray, sr: int, depth_db: float, above_db: float = 12.0, hold_s: float = 0.15) -> np.ndarray:
    """Quiet the gaps, not the voice: frames more than `above_db` over the room floor (speech, breaths, moans) and
    `hold_s` around them pass untouched; everything else drops `depth_db`. Denoising alone left a steady hiss in
    every pause on noisy references (heard as "white noise monotonics"); a dry reference sits at ~-94 dB and
    sounds silent between phrases -- this makes the others do the same. Gain moves smoothly (10 ms up, 120 ms
    down) and the hold reaches past word edges, so onsets and soft tails are never clipped."""
    h = int(0.01 * sr); n = len(audio) // h
    if n < 10: return audio
    db = 20 * np.log10(np.sqrt((audio[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    act = db > np.percentile(db, 10) + above_db
    k = int(hold_s / 0.01)
    act = np.convolve(act.astype(float), np.ones(2 * k + 1), "same") > 0
    tgt = np.where(act, 1.0, 10 ** (-depth_db / 20))
    g, up, down = np.empty(n), 1 - np.exp(-1 / 1.0), 1 - np.exp(-1 / 12.0)    # one-pole, per 10 ms frame
    cur = tgt[0]
    for i in range(n):
        cur += (tgt[i] - cur) * (up if tgt[i] > cur else down); g[i] = cur
    gain = np.interp(np.arange(len(audio)), np.arange(n) * h + h / 2, g)
    return (audio * gain).astype(np.float32)


def denoise(wav: np.ndarray, sr: int, strength: float) -> np.ndarray:
    """Stationary spectral gating of the finished piece: the clone copies the reference room's noise floor (~27 dB
    under the voice, where clean ASMR has 40+). strength 0.8 lowers the floor ~14 dB with transcripts unchanged
    (measured); stronger starts eating whispered speech."""
    import noisereduce as nr
    return nr.reduce_noise(y=wav, sr=sr, stationary=True, prop_decrease=strength, n_fft=1024,
                           freq_mask_smooth_hz=300, time_mask_smooth_ms=60).astype(np.float32)

def xfade(a: np.ndarray, b: np.ndarray, sr: int, ms: float = 30) -> np.ndarray:
    """Equal-power crossfade join: no click, no level dip at the seam."""
    n = min(int(ms / 1000 * sr), len(a) // 2, len(b) // 2)
    if n <= 0: return np.concatenate([a, b])
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    return np.concatenate([a[:-n], a[-n:] * np.cos(t) + b[:n] * np.sin(t), b[n:]]).astype(np.float32)


def quiet_run(wav: np.ndarray, sr: int, t0: float, t1: float, floor_db: float, above: float = 10.0) -> tuple[float, float]:
    """Longest stretch inside [t0, t1] within `above` dB of the room floor, as (seconds, centre): the pause a
    listener actually hears between two words -- measured, because Whisper's word edges are 100-200 ms loose."""
    h = int(0.01 * sr); a, b = int(max(0.0, t0) * sr), int(t1 * sr)
    seg = wav[a:b][: (b - a) // h * h]
    if len(seg) < h: return 0.0, (t0 + t1) / 2
    db = 20 * np.log10(np.sqrt((seg.reshape(-1, h) ** 2).mean(1)) + 1e-9)
    best, cur, end = 0, 0, 0
    for i, q in enumerate(db < floor_db + above):
        cur = cur + 1 if q else 0
        if cur > best: best, end = cur, i + 1
    return best * 0.01, (a / sr) + (end - best / 2) * 0.01


def local_level(wav: np.ndarray, sr: int, t: float, span: float = 0.05) -> float:
    seg = wav[max(0, int((t - span) * sr)): int((t + span) * sr)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def floor_level(wav: np.ndarray, sr: int) -> float:
    """The room's own level: RMS of the 10th-percentile 10 ms frame."""
    h = int(0.01 * sr); fr = wav[: len(wav) // h * h].reshape(-1, h)
    return float(np.percentile(np.sqrt((fr ** 2).mean(1)), 10)) if len(fr) else 0.0


def tone_level(wav: np.ndarray, sr: int, *ts: float) -> float:
    """Level for inserted room tone: the level around the cut, but never more than 3 dB over the room floor --
    a cut next to a word's tail must not turn the pause into a hiss at the tail's level."""
    return min([local_level(wav, sr, t) for t in ts] + [1.41 * floor_level(wav, sr)])


BREATH_CUES = [(re.compile(r"\b(breathe in|breathing in|inhale|take a (deep |slow )?breath)\b", re.I), 4.0),
               (re.compile(r"^\W*((and|now|slowly|then)\s+)*in\W*$", re.I), 4.0),               # "In..."
               (re.compile(r"\bhold (it|that|your breath)\b|^\W*hold\W*$", re.I), 3.0),
               (re.compile(r"\b(breathe out|breathing out|exhale|let it (all )?(out|go))\b", re.I), 5.0),
               (re.compile(r"^\W*((and|now|slowly|then)\s+)*out\W*$", re.I), 5.0)]              # "and out."
CLAUSE_END = re.compile(r"(\.\.\.|…|[.!?,;:])[\"')]*$")
SENT_END = re.compile(r"(\.\.\.|…|[.!?])[\"')]*$")
ELLIPSIS = re.compile(r"(\.\.\.|…)[\"')]*$")


def gap_plan(text: str, sentence_pause: float, ellipsis_pause: float, breath_cues: bool) -> dict:
    """What each gap between script words must hold, keyed (word before, word after) by script-word index:
    {"min": seconds from one word's end to the next word's start, "event": a vocal event tag sits there}.
    Sources: [pause Ns] tags, sentence ends, ellipses, and breathing cues (~4 s to inhale, ~3 s to hold, ~5 s
    to release after the clause that asks for it). None of these are written into the model's text: long
    pause tags invite the model to murmur through them, so the quiet is made here, out of room tone."""
    gaps, k, clause = {}, 0, []
    for tok in re.findall(r"\[[^\]]*\]|[^\s\[]+", text):
        m = TAG.fullmatch(tok)
        if m:
            g = gaps.setdefault((k - 1, k), {"min": 0.0, "event": False})
            if m.group(1) == "pause": g["min"] = max(g["min"], float(m.group(2) or 0.0))
            else: g["event"] = True
            continue
        if not _norm(tok): continue
        k += 1; clause.append(tok)
        if not CLAUSE_END.search(tok): continue
        c, clause = " ".join(clause), []
        secs = next((s for rx, s in BREATH_CUES if rx.search(c)), 0.0) if breath_cues else 0.0
        if ELLIPSIS.search(tok): secs = max(secs, ellipsis_pause)
        elif SENT_END.search(tok): secs = max(secs, sentence_pause)
        if secs:
            g = gaps.setdefault((k - 1, k), {"min": 0.0, "event": False}); g["min"] = max(g["min"], secs)
    return gaps


_ALIGN = None


def word_spans(wav: np.ndarray, sr: int, words: list[str]) -> list[tuple[float, float] | None]:
    """Precise (start, end) of each heard word by wav2vec2 CTC forced alignment (20 ms frames). Whisper's own
    word times are 100-500 ms loose -- it glued "thing" onto "just" and split "any|thing" -- so every edit is
    placed against these spans instead. Words with no letters get None."""
    global _ALIGN
    import librosa, torchaudio, torchaudio.functional as AF
    if _ALIGN is None:
        b = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        _ALIGN = (b.get_model().eval(), {c: i for i, c in enumerate(b.get_labels())})
    m, idx = _ALIGN
    toks = [re.sub(r"[^A-Z']", "", w.upper()) for w in words]
    keep = [i for i, t in enumerate(toks) if t]
    out: list = [None] * len(words)
    if not keep: return out
    x16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    with torch.inference_mode():
        em = torch.log_softmax(m(torch.from_numpy(x16).float()[None])[0], -1)
    seq = [idx[c] for c in "|".join(toks[i] for i in keep)]
    if len(seq) >= em.shape[1]: return out
    ali, sc = AF.forced_align(em, torch.tensor([seq], dtype=torch.int32), blank=0)
    spans = AF.merge_tokens(ali[0], sc[0].exp())
    r, k = len(x16) / 16000 / em.shape[1], 0
    for i in keep:
        n = len(toks[i]); sp = spans[k:k + n]; k += n + 1               # +1: the "|" between words
        if sp: out[i] = (sp[0].start * r, sp[-1].end * r)
    return out


def align_trim(wav: np.ndarray, sr: int, text: str, lookahead: list[str], event_pause: float = 0.7,
               sentence_pause: float = 0.9, ellipsis_pause: float = 1.4, breath_cues: bool = True,
               edit_log: list | None = None) -> tuple[np.ndarray, str, float, float]:
    """Trim the chunk to its own words and give every gap the quiet it needs -- never cutting into a word.

    Words are heard by Whisper and timed by forced alignment (word_spans). Every edit lies strictly between
    two aligned words:
    * start/end cuts at the quietest point just outside the first word / between the last word and the
      look-ahead (the chunk was generated with the next chunk's first words appended, so its last word is
      spoken fully); a chunk that opens with a tag keeps its opening
    * pause gaps (tags, sentence ends, breathing cues -- gap_plan): the quiet that is actually there is measured
      and topped up inside the longest quiet run between the words; words run together get the pause at their
      boundary. Words heard there that the script doesn't say (murmurs) are replaced by room tone
    * a vocal event gap gets `event_pause` of quiet at the start of the last quiet stretch before the next word,
      i.e. after the event has finished, so a moan, "mhm" or laugh is never split
    Returns (audio, note, quiet before the first word, quiet after the last word) -- the join needs the last two."""
    global _ASR
    import difflib, librosa
    from faster_whisper import WhisperModel
    if _ASR is None: _ASR = WhisperModel("small.en", device="cpu", compute_type="int8")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    words = [w for seg in _ASR.transcribe(w16, language="en", word_timestamps=True, vad_filter=False)[0] for w in (seg.words or [])]
    if not words: return wav, "no words heard", 0.0, 0.0
    sp = word_spans(wav, sr, [w.word for w in words])
    st = [s[0] if s else w.start for s, w in zip(sp, words)]
    en = [s[1] if s else w.end for s, w in zip(sp, words)]
    real = [_norm(t) for t in TAG.sub(" ", text).split() if _norm(t)]
    la = [_norm(t) for t in lookahead if _norm(t)]
    sm = difflib.SequenceMatcher(a=real + la, b=[_norm(w.word) for w in words], autojunk=False)
    s2h = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size): s2h[blk.a + k] = blk.b + k
    jw = max((s2h[i] for i in range(len(real)) if i in s2h), default=None)
    jl = min((s2h[i] for i in range(len(real), len(real) + len(la)) if i in s2h), default=None)
    if jw is None: return wav, "no script words matched", 0.0, 0.0
    dropped = len(real) - 1 - max(i for i in range(len(real)) if i in s2h)
    dur = len(wav) / sr

    h = int(0.01 * sr); fdb = 20 * np.log10(np.sqrt((wav[: len(wav) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
    quiet = fdb < float(np.percentile(fdb, 10)) + 10.0
    loud = fdb > float(np.percentile(fdb, 95)) - 15.0               # speech-level sound

    def after_speech(t0: float, t1: float) -> float:
        """End of the last speech-level frame in [t0, t1] (t0 if none)."""
        a, b = max(0, int(t0 * 100)), min(len(loud), int(t1 * 100))
        idx = np.where(loud[a:b])[0]
        return (a + idx[-1] + 1) / 100 if len(idx) else t0

    def before_speech(t0: float, t1: float) -> float:
        """Start of the first speech-level frame in [t0, t1] (t1 if none)."""
        a, b = max(0, int(t0 * 100)), min(len(loud), int(t1 * 100))
        idx = np.where(loud[a:b])[0]
        return (a + idx[0]) / 100 if len(idx) else t1

    def word_edges(t_prev: float, t_next: float) -> tuple[float, float]:
        """Where the words around a gap really end/start. The aligner clips words: it ended "relax" at "rela"
        (the "x" sits after it) and started "If" at the "f" (the "I" sits before it). Sound within 0.25 s after
        a word is its tail, within 0.3 s before the next word its head -- a pause never goes inside either."""
        tail = after_speech(t_prev, min(t_prev + 0.25, t_next))
        return tail, max(tail, before_speech(max(tail, t_next - 0.30), t_next))

    def strays(t0: float, t1: float) -> list[tuple[float, float]]:
        """Short (< 0.3 s) speech-level sounds inside [t0, t1] that Whisper heard no word in: fragments the model
        drops into pauses, which sound like a word cut off ("bu-") once the pause around them is stretched."""
        a, b = max(0, int(t0 * 100)), min(len(loud), int(t1 * 100))
        out, i = [], a
        while i < b:
            if loud[i]:
                j = i
                while j < b and (loud[j] or (j + 5 < b and loud[j:j + 5].any())): j += 1
                if j - i < 30: out.append((i / 100, j / 100))
                i = j
            else: i += 1
        return out

    def runs(t0: float, t1: float) -> list[tuple[float, float]]:
        """Quiet stretches (>= 30 ms) inside [t0, t1], as (start, end) seconds."""
        a, b = max(0, int(t0 * 100)), min(len(quiet), int(t1 * 100))
        out, i = [], a
        while i < b:
            if quiet[i]:
                j = i
                while j < b and quiet[j]: j += 1
                if j - i >= 3: out.append((i / 100, j / 100))
                i = j
            else: i += 1
        return out

    gaps = gap_plan(text, sentence_pause, ellipsis_pause, breath_cues)
    w0 = before_speech(max(0.0, st[0] - 0.30), st[0])                  # include a head the aligner left out
    s0 = 0.0 if re.match(r"\s*\[", text) else (quietest(wav, sr, w0 - 0.45, w0 - 0.02) or max(0.0, w0 - 0.05))
    hi = (st[jl] - 0.02) if (jl is not None and jl > jw) else min(dur, en[jw] + 0.5)
    end_w = after_speech(en[jw], min(hi, en[jw] + 0.25))              # where the last word's sound really ends
    tail = [r for r in runs(end_w, hi) if r[1] - r[0] >= 0.15]       # a real pause, not a stop inside the word
    s1 = min(dur, tail[0][0] + min(0.08, (tail[0][1] - tail[0][0]) / 2) if tail else hi)

    # edits: (cut_from, cut_to, room-tone seconds, crossfade ms); cut_from == cut_to is a pure insert
    edits, cleaned, missed = [], 0, 0
    for (prev_k, next_k), g in sorted(gaps.items()):
        if next_k not in s2h or next_k >= len(real) or (prev_k >= 0 and prev_k not in s2h): continue
        t_next = st[s2h[next_k]]
        t_prev = en[s2h[prev_k]] if prev_k >= 0 else s0
        if t_next <= t_prev: missed += 1; continue
        _, head = word_edges(t_prev, t_next)
        q = runs(t_prev, head)
        if g["event"]:
            need = max(0.0, g["min"] - (t_next - t_prev)) + event_pause
            last = q[-1] if q and q[-1][1] >= head - 0.05 else None            # quiet right up to the next word
            if last:
                edits.append((last[0] + min(0.03, (last[1] - last[0]) / 2),) * 2 + (need, 60)); continue
            # the event runs into the next word: the pause goes before the event, in a real pause (>= 150 ms; a
            # shorter dip can be inside a drawn-out word), else just before the next word's onset
            first = next((r for r in q if r[1] - r[0] >= 0.15), None)
            at = first[0] + 0.03 if first else head - 0.02
            edits.append((at, at, need, 60 if first else 15)); continue
        junk = words[(s2h[prev_k] if prev_k >= 0 else -1) + 1: s2h[next_k]]
        if junk:
            # long training pauses often held unlabelled mouth sounds, so the model can murmur through a pause
            j0, j1 = st[s2h[next_k] - len(junk)], en[s2h[next_k] - 1]
            a_ = quietest(wav, sr, t_prev, j0) if j0 - t_prev >= 0.04 else None
            b_ = quietest(wav, sr, j1, t_next) if t_next - j1 >= 0.04 else None
            a_, b_ = a_ or max(t_prev, j0 - 0.01), b_ or min(t_next, j1 + 0.01)
            if t_prev <= a_ < b_ <= t_next:
                edits.append((a_, b_, max(0.3, g["min"] - (a_ - t_prev) - (t_next - b_)), 30)); cleaned += 1
                continue
        tail, head = word_edges(t_prev, t_next)
        for a_, b_ in strays(tail + 0.03, head - 0.03):                   # fragments in the pause -> room tone
            a_, b_ = max(tail, a_ - 0.03), min(head, b_ + 0.05)
            edits.append((a_, b_, b_ - a_, 30)); quiet[int(a_ * 100):int(b_ * 100)] = True; cleaned += 1
        q = runs(tail, head)
        heard = max((b - a for a, b in q), default=0.0)
        need = g["min"] - heard
        if need <= 0.08: continue
        if q:
            a, b = max(q, key=lambda r: r[1] - r[0])
            at = quietest(wav, sr, a, b, win=min(0.04, b - a)) or (a + b) / 2
            if any(x0 <= at <= x1 for x0, x1, _, _ in edits if x1 > x0): at = a + 0.02   # not inside a stray cut
            edits.append((at, at, need, 60))
        else:
            edits.append((max(tail, head - 0.02),) * 2 + (need, 15))         # run together: just before the next word

    out, last, added = None, s0, 0.0
    for a0, b0, fill, ms in sorted(edits):
        if not (s0 < a0 <= b0 < s1) or a0 < last: missed += 1; continue      # outside the trim, or overlapping
        if edit_log is not None:
            before = [w.word.strip() for w, e in zip(words, en) if e <= a0 + 0.01][-1:]
            after = [w.word.strip() for w, b in zip(words, st) if b >= b0 - 0.01][:1]
            edit_log.append({"at": round(a0, 3), "cut_to": round(b0, 3), "room_tone_s": round(fill, 2),
                             "between": before + after})
        seg = wav[int(last * sr): int(a0 * sr)]
        gap = room_noise(wav, sr, fill, tone_level(wav, sr, a0, b0))
        out = seg if out is None else xfade(out, seg, sr, ms=ms)
        out = xfade(out, gap, sr, ms=ms); last = b0; added += fill - (b0 - a0)
    seg = wav[int(last * sr): int(s1 * sr)]
    out = seg if out is None else xfade(out, seg, sr, ms=30)
    note = "ok" if dropped <= 0 else f"model dropped the last {dropped} word(s)"
    if cleaned: note += f", {cleaned} murmured pause(s) cleaned"
    if missed: note += f", {missed} pause(s) not placeable"
    lead = 0.0 if s0 == 0.0 else max(0.0, w0 - s0)
    return (out.astype(np.float32), note + (f", {added:+.1f}s pause adjustment" if abs(added) > 0.05 else ""),
            lead, max(0.0, s1 - end_w))

FILLERS = {"mm", "mmm", "mmmm", "hmm", "hm", "mhm", "oh", "ohh", "ah", "ahh", "uh", "um", "huh"}


_ASR_M = None


def script_errors(wav: np.ndarray, sr: int, text: str, careful: bool = False) -> int:
    """Word edits between the script and what is heard in a trimmed chunk; vocal-event noises Whisper writes as
    "mm" / "oh" are not counted (they are the events), repeated or invented words are."""
    import difflib, librosa
    global _ASR, _ASR_M
    from faster_whisper import WhisperModel
    if _ASR is None: _ASR = WhisperModel("small.en", device="cpu", compute_type="int8")
    # moan chunks are heard with medium.en: small.en writes a laugh off as silence, medium hears "hahaha"
    if careful and _ASR_M is None: _ASR_M = WhisperModel("medium.en", device="cpu", compute_type="int8")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    heard = [_norm(w.word) for seg in (_ASR_M if careful else _ASR).transcribe(w16, language="en", word_timestamps=True)[0]
             for w in (seg.words or [])]
    heard = [w for w in heard if w and w not in FILLERS]
    real = [_norm(t) for t in TAG.sub(" ", text).split() if _norm(t)]
    # a laugh the script doesn't ask for is worse than a wrong word: the user heard v1.1 laughs as fake
    laughs = 0 if re.search(r"laugh|\bha", text, re.I) else sum(bool(re.fullmatch(r"(ha|he|hah|heh)+h?", w)) for w in heard)
    heard = [w for w in heard if not re.fullmatch(r"(ha|he|hah|heh)+h?", w)]
    ops = difflib.SequenceMatcher(a=real, b=heard, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal") + 3 * laughs


def end_pause(text: str, sentence_pause: float, ellipsis_pause: float, breath_cues: bool) -> float:
    """The quiet a chunk's last word asks for (sentence end, ellipsis, breathing cue): kept across the join."""
    n = len([t for t in TAG.sub(" ", text).split() if _norm(t)])
    g = gap_plan(text, sentence_pause, ellipsis_pause, breath_cues).get((n - 1, n))
    return g["min"] if g else 0.0


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
    ap.add_argument("--ref", help="reference voice clip (6-12 s of clean speech)")
    ap.add_argument("--script", help="text file, or the script itself")
    ap.add_argument("--restitch", type=Path, default=None,
                    help="a <out>.takes dir from an earlier render: re-stitch its saved takes, no model, no new takes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="aoxo/text2asmr-t3-v2", help="HF repo or local dir; '' for the base model")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-chunk-s", type=float, default=18.0)
    ap.add_argument("--temperature", type=float, default=0.65, help="lower = fewer garbled/odd words")
    ap.add_argument("--cfg", type=float, default=0.3, help="lower = slower, more deliberate pacing")
    ap.add_argument("--exaggeration", type=float, default=0.35, help="lower = gentler, calmer delivery")
    ap.add_argument("--rep-penalty", type=float, default=1.2, help="T3 repetition penalty (lower loops: measured)")
    ap.add_argument("--denoise", type=float, default=0.8, help="noise-reduction strength (0 = off)")
    ap.add_argument("--gate", type=float, default=30.0, help="dB to pull the gaps between phrases down by (0 = off)")
    ap.add_argument("--speed", type=float, default=1.0, help="tempo of the finished piece, pitch kept (0.9 = 10%% slower)")
    ap.add_argument("--moan-exaggeration", type=float, default=None, help="exaggeration for chunks with moans (default: --exaggeration)")
    ap.add_argument("--moan-temperature", type=float, default=None, help="temperature for chunks with moans (default: --temperature)")
    ap.add_argument("--event-pause", type=float, default=0.7, help="quiet seconds after each vocal event tag")
    ap.add_argument("--sentence-pause", type=float, default=0.9, help="minimum quiet between sentences (s)")
    ap.add_argument("--ellipsis-pause", type=float, default=1.4, help="minimum quiet after '...' (s)")
    ap.add_argument("--no-breath-cues", action="store_true", help="don't hold inhale/hold/exhale pauses after breathing cues")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", type=int, default=3, help="generate N takes per chunk, keep the one closest to the script")
    ap.add_argument("--max-candidates", type=int, default=6, help="keep trying up to this many while the best take has >= 2 word errors")
    ap.add_argument("--no-align-trim", action="store_true", help="skip word-aligned trimming of chunk edges")
    ap.add_argument("--gap-s", type=float, default=0.45, help="quiet between chunks when no pause is asked for")
    a = ap.parse_args()
    import json, soundfile as sf
    pauses = dict(sentence_pause=a.sentence_pause, ellipsis_pause=a.ellipsis_pause, breath_cues=not a.no_breath_cues)
    if a.restitch:                                           # re-stitch saved takes with the current post-processing
        meta = json.loads((a.restitch / "takes.json").read_text())
        chunks, pieces = [c["text"] for c in meta["chunks"]], []
        for i, c in enumerate(meta["chunks"]):
            raw, sr = sf.read(a.restitch / c["file"], dtype="float32")
            if a.no_align_trim: pieces.append((raw, 0.0, 0.0)); continue
            wav, note, lead, trail = align_trim(raw, sr, c["text"], c["lookahead"], a.event_pause, **pauses)
            log(f"chunk {i + 1}/{len(chunks)} (saved take {c['take'] + 1}): {len(raw) / sr:.1f} s -> {len(wav) / sr:.1f} s ({note})")
            pieces.append((wav, lead, trail))
        import perth
        finish(assemble(pieces, chunks, sr, a, pauses), sr, a, perth.PerthImplicitWatermarker())
        return 0
    if not (a.ref and a.script): ap.error("--ref and --script are required unless --restitch is given")
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
    pieces, sr = [], model.sr
    # every chosen take is kept raw, so a take worth keeping survives later changes to the post-processing
    takes_dir = Path(str(Path(a.out).with_suffix("")) + ".takes"); takes_dir.mkdir(parents=True, exist_ok=True)
    meta = {"script": script, "ref": a.ref, "adapter": a.adapter, "seed": a.seed, "chunks": []}
    TAIL = ["Okay", "then."]                                  # look-ahead for the final chunk; always cut off
    for i, text in enumerate(chunks):
        lookahead = (TAG.sub(" ", chunks[i + 1]).split()[:3] if i + 1 < len(chunks) else TAIL)
        gen_text = text if a.no_align_trim else f"{text} {' '.join(lookahead)}"
        moan = bool(MOAN.search(text))                    # calm narration settings flatten moans into a monotone
        exag = a.moan_exaggeration if moan and a.moan_exaggeration is not None else a.exaggeration
        temp = a.moan_temperature if moan and a.moan_temperature is not None else a.temperature
        tt = model.tokenizer.text_to_tokens(punc_norm(gen_text)).to(model.device)
        if a.cfg > 0: tt = torch.cat([tt, tt], dim=0)
        tt = F.pad(F.pad(tt, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token)
        cond = T3Cond(speaker_emb=base.speaker_emb, cond_prompt_speech_tokens=prompt,
                      emotion_adv=exag * torch.ones(1, 1, 1)).to(device=model.device)
        # best of N: the model sometimes repeats or inserts words ("Mmm, good. Good."); each candidate is
        # transcribed and the one closest to the script wins (ties keep the earliest, so candidate 0 -- the
        # plain --seed render -- stands unless another is strictly better)
        best, t0 = None, time.time()
        for j in range(max(1, a.candidates, a.max_candidates)):
            if j: torch.manual_seed(a.seed + 7919 * j + 104729 * i)   # take 0 continues the --seed stream exactly
            with torch.inference_mode():
                st = model.t3.inference(t3_cond=cond, text_tokens=tt, max_new_tokens=520, temperature=temp,
                                        cfg_weight=a.cfg, repetition_penalty=a.rep_penalty, min_p=0.05, top_p=1.0)[0]
                st = drop_invalid_tokens(st); st = st[st < 6561].to(model.device)
                raw, _ = model.s3gen.inference(speech_tokens=st, ref_dict=model.conds.gen)
            raw = raw.squeeze(0).detach().cpu().numpy()
            wav, note, lead, new_trail, edits = raw, "untrimmed", 0.0, 0.0, []
            if not a.no_align_trim:
                wav, note, lead, new_trail = align_trim(raw, sr, text, lookahead, a.event_pause, **pauses, edit_log=edits)
            # score what the listener will hear: denoised (a laugh only stood out to Whisper once the hiss was gone)
            heard = denoise(wav, sr, a.denoise) if a.denoise > 0 else wav
            errs = script_errors(heard, sr, text, careful=moan) if a.candidates > 1 else 0
            if best is None or errs < best[0]: best = (errs, j, wav, note, lead, new_trail, raw, edits)
            if errs == 0 or (j + 1 >= a.candidates and best[0] <= 1): break
        errs, j, wav, note, lead, new_trail, raw, edits = best
        pick = f", candidate {j + 1} ({errs} word errors)" if a.candidates > 1 else ""
        log(f"chunk {i + 1}/{len(chunks)}: {len(raw) / sr:.1f} s -> {len(wav) / sr:.1f} s ({note}{', moan settings' if moan else ''}{pick}; "
            f"predicted {predicted_s(text):.1f} s) in {time.time() - t0:.0f} s | {text[:70]}")
        sf.write(takes_dir / f"chunk{i:02d}.wav", raw, sr, subtype="PCM_16")
        meta["chunks"].append({"file": f"chunk{i:02d}.wav", "text": text, "lookahead": lookahead, "take": j,
                               "word_errors": errs, "moan_settings": moan, "edits": edits})
        pieces.append((wav, lead, new_trail))
        # continuation prompt from the *trimmed* chunk, so a garbled last word never seeds the next chunk
        import librosa
        tail16 = librosa.resample(wav[-int(6 * sr):], orig_sr=sr, target_sr=16000)
        with torch.inference_mode():
            pt, pl = model.s3gen.tokenizer.forward([tail16])
        if int(pl[0]) >= PROMPT_TOKENS:
            prompt = pt[:, :int(pl[0])][:, -PROMPT_TOKENS:].long().to(model.device)
    (takes_dir / "takes.json").write_text(json.dumps(meta, indent=1))
    finish(assemble(pieces, chunks, sr, a, pauses), sr, a, model.watermarker)
    return 0


def assemble(pieces: list, chunks: list[str], sr: int, a, pauses: dict) -> np.ndarray:
    """Join trimmed chunks (wav, quiet before first word, quiet after last word) through room tone -- never
    digital silence. The quiet the previous chunk's last word asks for (sentence end, breathing cue) carries
    across the seam; the closing line's pause (e.g. 5 s to exhale after "and out.") is room tone fading out."""
    out = None
    for i, (wav, lead, trail) in enumerate(pieces):
        if out is None: out = wav
        else:
            ptrail = pieces[i - 1][2]
            want = 0.0 if re.match(r"\s*\[pause", chunks[i]) else end_pause(chunks[i - 1], **pauses)
            gap_s = max(0.15, want - ptrail - lead) if want else a.gap_s
            gap = room_noise(out, sr, gap_s, tone_level(out, sr, len(out) / sr - 0.06))
            out = xfade(xfade(out, gap, sr, ms=60), wav, sr, ms=60)
    if not a.no_align_trim:
        tail_s = max(3.0, min(5.0, end_pause(chunks[-1], **pauses))) - pieces[-1][2]   # room for a graceful fade-out
        if tail_s > 0.1:
            fade = (0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, int(tail_s * sr)))).astype(np.float32)
            out = xfade(out, room_noise(out, sr, tail_s, tone_level(out, sr, len(out) / sr - 0.06)) * fade, sr, ms=60)
    return out


def finish(audio: np.ndarray, sr: int, a, watermarker) -> None:
    import soundfile as sf
    if a.speed != 1.0: audio = tempo(audio, sr, a.speed)
    # denoise once, after stitching: inserted room tone and the real floor get the same gating, so the quiet
    # never switches texture between them (denoising chunks first made inserted pauses stand out)
    if a.denoise > 0: audio = denoise(audio, sr, a.denoise)
    if a.gate > 0: audio = gate(audio, sr, a.gate)
    audio = watermarker.apply_watermark(audio, sample_rate=sr)
    sf.write(a.out, audio, sr)
    log(f"wrote {a.out}: {len(audio) / sr:.1f} s")

if __name__ == "__main__":
    raise SystemExit(main())
