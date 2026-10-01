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


def decreak(x: np.ndarray, sr: int, thresh: float = 0.3, octave_gate: float = 0.65, passes: int = 2) -> tuple[np.ndarray, float]:
    """Repair vocal fry the clone copied from a creaky reference voice: period doubling, every other glottal cycle
    different, heard as a tremble/rattle. In those frames the pitch tracker reads about an octave below the
    speaker's normal pitch and the signal also repeats at half that period P; averaging each cycle with the one
    before it, (x(t) + x(t - P)) / 2, cancels the subharmonics and keeps the true harmonics. Only frames below
    `octave_gate` x the speaker's median pitch are touched (normal speech is never averaged), with overlap-add
    fades. Returns (audio, fraction of frames repaired in the first pass)."""
    import librosa
    hop, fl = int(0.01 * sr), int(0.04 * sr); first = None
    for _ in range(passes):
        f0, _, pv = librosa.pyin(x, fmin=60, fmax=500, sr=sr, frame_length=2048, hop_length=hop)
        clear = ~np.isnan(f0) & (pv > 0.5)
        if clear.sum() < 20: break
        med = float(np.median(f0[clear])); P = np.zeros(len(f0))
        for i, f in enumerate(f0):
            if np.isnan(f) or f > octave_gate * med: continue
            seg = x[max(0, i * hop - fl // 2): i * hop + fl // 2]
            if len(seg) < fl: continue
            seg = seg - seg.mean(); T0 = sr / f

            def r(lag):
                k = int(round(lag)); u, v = seg[:-k], seg[k:]
                return float(np.dot(u, v) / (np.sqrt(np.dot(u, u) * np.dot(v, v)) + 1e-12))
            if r(T0 / 2) > thresh and r(T0) > thresh: P[i] = T0 / 2
        if first is None: first = float((P > 0).mean())
        if not P.any(): break
        acc, w, win = np.zeros(len(x)), np.zeros(len(x)), np.hanning(2 * hop)
        for i in np.where(P > 0)[0]:
            a, b = max(0, i * hop - hop), min(len(x), i * hop + hop)
            k, fr = int(P[i]), P[i] - int(P[i]); d = np.arange(a, b) - k
            delayed = (1 - fr) * x[np.clip(d, 0, None)] + fr * x[np.clip(d - 1, 0, None)]
            acc[a:b] += 0.5 * (x[a:b] + delayed) * win[: b - a]; w[a:b] += win[: b - a]
        m = w > 1e-3; mix = np.clip(w, 0, 1); y = x.copy()
        y[m] = (1 - mix[m]) * x[m] + mix[m] * (acc[m] / w[m]); x = y.astype(np.float32)
    return x, first or 0.0


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


def align_trim(wav: np.ndarray, sr: int, text: str, lookahead: list[str], event_pause: float = 0.7,
               sentence_pause: float = 0.9, ellipsis_pause: float = 1.4, breath_cues: bool = True
               ) -> tuple[np.ndarray, str, float, float]:
    """Trim the chunk to its own words and give every gap the quiet it needs, with every edit in silence.

    * look-ahead: the chunk was generated with the next chunk's first words appended, so its real last word is
      spoken fully; the end cut goes at the quietest point between that word and the look-ahead
    * the start cut goes at the quietest point before the first word -- or nowhere, if the chunk opens with a tag
      (trimming to the first word cut the opening breath away)
    * each gap is stretched to its gap_plan() minimum; a vocal event also gets `event_pause` s of quiet after it,
      inserted where the event has decayed. Words heard inside a pause-only gap that the script doesn't say
      (murmurs, stray "mm"s) are replaced. All inserted quiet is room tone matched to the level around it.
    Returns (audio, note, quiet before the first word, quiet after the last word) -- the join needs the last two."""
    global _ASR
    import difflib, librosa
    from faster_whisper import WhisperModel
    if _ASR is None: _ASR = WhisperModel("small.en", device="cpu", compute_type="int8")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    words = [w for seg in _ASR.transcribe(w16, language="en", word_timestamps=True, vad_filter=False)[0] for w in (seg.words or [])]
    if not words: return wav, "no words heard", 0.0, 0.0
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

    gaps = gap_plan(text, sentence_pause, ellipsis_pause, breath_cues)
    if re.match(r"\s*\[", text):
        s0 = 0.0
    else:
        w0 = words[0].start
        s0 = quietest(wav, sr, w0 - 0.45, w0 - 0.08) or max(0.0, w0 - 0.25)
    end_w = words[jw].end
    hi = (words[jl].start - 0.05) if (jl is not None and jl > jw) else min(dur, end_w + 0.5)
    s1 = min(dur, quietest(wav, sr, end_w + 0.05, hi) or end_w + 0.25)

    # edits: (cut_from, cut_to, room-tone seconds); cut_from == cut_to is a pure insert
    edits, cleaned, missed = [], 0, 0
    h = int(0.01 * sr); fdb = 20 * np.log10(np.sqrt((wav[: len(wav) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
    floor_db = float(np.percentile(fdb, 10))
    for (prev_k, next_k), g in sorted(gaps.items()):
        if next_k not in s2h or next_k >= len(real) or (prev_k >= 0 and prev_k not in s2h): continue
        t_next = words[s2h[next_k]].start
        t_prev = words[s2h[prev_k]].end if prev_k >= 0 else s0
        if g["event"]:
            need, lo = max(0.0, g["min"] - (t_next - t_prev)) + event_pause, (t_prev + t_next) / 2   # after it decays
        else:
            # long training pauses often held unlabelled mouth sounds, so the model can murmur through a pause;
            # anything heard as words inside it that the script doesn't say is replaced by room tone
            junk = words[(s2h[prev_k] if prev_k >= 0 else -1) + 1: s2h[next_k]]
            if junk:
                a0 = quietest(wav, sr, t_prev + 0.05, junk[0].start - 0.03) or max(t_prev + 0.05, junk[0].start - 0.1)
                b0 = quietest(wav, sr, junk[-1].end + 0.03, t_next - 0.12) or min(t_next - 0.12, junk[-1].end + 0.1)
                if s0 < a0 < b0 < s1:
                    edits.append((a0, b0, max(0.3, g["min"] - (a0 - t_prev) - (t_next - b0)))); cleaned += 1
                    continue
            # the pause that is actually there, measured from the audio around the boundary
            heard, centre = quiet_run(wav, sr, t_prev - 0.15, t_next + 0.15, floor_db)
            need = g["min"] - heard
            if need > 0.08 and heard >= 0.04:
                at = quietest(wav, sr, centre - heard / 2, centre + heard / 2, win=min(0.04, heard))
                if at is not None and s0 < at < s1: edits.append((at, at, need)); continue
            lo = t_prev + 0.08
        at = quietest(wav, sr, lo, t_next - 0.12)
        if at is None and need > 0.08:                    # words run together: Whisper's edges are 100-200 ms off,
            mid = (t_prev + t_next) / 2                   # so look for the quiet point around the boundary
            at = quietest(wav, sr, min(lo, mid - 0.15), max(t_next - 0.12, mid + 0.15), win=0.03)
        if need > 0.08 and at is not None and s0 < at < s1: edits.append((at, at, need))
        elif need > 0.08: missed += 1

    out, last, added = None, s0, 0.0
    for a0, b0, fill in sorted(edits):
        seg = wav[int(last * sr): int(a0 * sr)]
        gap = room_noise(wav, sr, fill, tone_level(wav, sr, a0, b0))
        out = seg if out is None else xfade(out, seg, sr, ms=60)
        out = xfade(out, gap, sr, ms=60); last = b0; added += fill - (b0 - a0)
    seg = wav[int(last * sr): int(s1 * sr)]
    out = seg if out is None else xfade(out, seg, sr, ms=60)
    note = "ok" if dropped <= 0 else f"model dropped the last {dropped} word(s)"
    if cleaned: note += f", {cleaned} murmured pause(s) cleaned"
    if missed: note += f", {missed} pause(s) not placeable"
    lead = 0.0 if s0 == 0.0 else max(0.0, words[0].start - s0)
    return (out.astype(np.float32), note + (f", {added:+.1f}s pause adjustment" if abs(added) > 0.05 else ""),
            lead, max(0.0, s1 - end_w))


FILLERS = {"mm", "mmm", "mmmm", "hmm", "hm", "mhm", "oh", "ohh", "ah", "ahh", "uh", "um", "huh"}


def script_errors(wav: np.ndarray, sr: int, text: str) -> int:
    """Word edits between the script and what is heard in a trimmed chunk; vocal-event noises Whisper writes as
    "mm" / "oh" are not counted (they are the events), repeated or invented words are."""
    import difflib, librosa
    global _ASR
    if _ASR is None:
        from faster_whisper import WhisperModel
        _ASR = WhisperModel("small.en", device="cpu", compute_type="int8")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    heard = [_norm(w.word) for seg in _ASR.transcribe(w16, language="en", word_timestamps=True)[0] for w in (seg.words or [])]
    heard = [w for w in heard if w and w not in FILLERS]
    real = [_norm(t) for t in TAG.sub(" ", text).split() if _norm(t)]
    ops = difflib.SequenceMatcher(a=real, b=heard, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")


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
    ap.add_argument("--ref", required=True, help="reference voice clip (6-12 s of clean speech)")
    ap.add_argument("--script", required=True, help="text file, or the script itself")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="aoxo/text2asmr-t3-v2", help="HF repo or local dir; '' for the base model")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-chunk-s", type=float, default=18.0)
    ap.add_argument("--temperature", type=float, default=0.65, help="lower = fewer garbled/odd words")
    ap.add_argument("--cfg", type=float, default=0.3, help="lower = slower, more deliberate pacing")
    ap.add_argument("--exaggeration", type=float, default=0.35, help="lower = gentler, calmer delivery")
    ap.add_argument("--rep-penalty", type=float, default=1.2, help="T3 repetition penalty (lower loops: measured)")
    ap.add_argument("--no-decreak", action="store_true", help="keep vocal-fry tremble cloned from the reference")
    ap.add_argument("--moan-exaggeration", type=float, default=None, help="exaggeration for chunks with moans (default: --exaggeration)")
    ap.add_argument("--moan-temperature", type=float, default=None, help="temperature for chunks with moans (default: --temperature)")
    ap.add_argument("--event-pause", type=float, default=0.7, help="quiet seconds after each vocal event tag")
    ap.add_argument("--sentence-pause", type=float, default=0.9, help="minimum quiet between sentences (s)")
    ap.add_argument("--ellipsis-pause", type=float, default=1.4, help="minimum quiet after '...' (s)")
    ap.add_argument("--no-breath-cues", action="store_true", help="don't hold inhale/hold/exhale pauses after breathing cues")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", type=int, default=3, help="generate N takes per chunk, keep the one closest to the script")
    ap.add_argument("--no-align-trim", action="store_true", help="skip word-aligned trimming of chunk edges")
    ap.add_argument("--gap-s", type=float, default=0.45, help="quiet between chunks when no pause is asked for")
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
    pauses = dict(sentence_pause=a.sentence_pause, ellipsis_pause=a.ellipsis_pause, breath_cues=not a.no_breath_cues)
    trail = 0.0
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
        for j in range(max(1, a.candidates)):
            if j: torch.manual_seed(a.seed + 7919 * j + 104729 * i)   # take 0 continues the --seed stream exactly
            with torch.inference_mode():
                st = model.t3.inference(t3_cond=cond, text_tokens=tt, max_new_tokens=520, temperature=temp,
                                        cfg_weight=a.cfg, repetition_penalty=a.rep_penalty, min_p=0.05, top_p=1.0)[0]
                st = drop_invalid_tokens(st); st = st[st < 6561].to(model.device)
                wav, _ = model.s3gen.inference(speech_tokens=st, ref_dict=model.conds.gen)
            wav = wav.squeeze(0).detach().cpu().numpy()
            raw_s = len(wav) / sr
            note, lead, new_trail = "untrimmed", 0.0, 0.0
            if not a.no_align_trim:
                wav, note, lead, new_trail = align_trim(wav, sr, text, lookahead, a.event_pause, **pauses)
            errs = script_errors(wav, sr, text) if a.candidates > 1 else 0
            if best is None or errs < best[0]: best = (errs, j, wav, note, lead, new_trail, raw_s)
            if errs == 0: break
        errs, j, wav, note, lead, new_trail, raw_s = best
        pick = f", candidate {j + 1} ({errs} word errors)" if a.candidates > 1 else ""
        log(f"chunk {i + 1}/{len(chunks)}: {raw_s:.1f} s -> {len(wav) / sr:.1f} s ({note}{', moan settings' if moan else ''}{pick}; "
            f"predicted {predicted_s(text):.1f} s) in {time.time() - t0:.0f} s | {text[:70]}")
        # join through the previous chunk's own room tone at its closing level -- never digital silence; the
        # quiet the previous chunk's last word asks for (sentence end, breathing cue) carries across the seam
        if not out: out = [wav]
        else:
            prev = out[-1]
            want = 0.0 if re.match(r"\s*\[pause", text) else end_pause(chunks[i - 1], **pauses)
            gap_s = max(0.15, want - trail - lead) if want else a.gap_s
            gap = room_noise(prev, sr, gap_s, tone_level(prev, sr, len(prev) / sr - 0.06))
            out[-1] = xfade(xfade(prev, gap, sr, ms=60), wav, sr, ms=60)
        if not a.no_align_trim: trail = new_trail
        # continuation prompt from the *trimmed* tail, so a garbled last word never seeds the next chunk
        import librosa
        tail16 = librosa.resample(wav[-int(6 * sr):], orig_sr=sr, target_sr=16000)
        with torch.inference_mode():
            pt, pl = model.s3gen.tokenizer.forward([tail16])
        if int(pl[0]) >= PROMPT_TOKENS:
            prompt = pt[:, :int(pl[0])][:, -PROMPT_TOKENS:].long().to(model.device)
    audio = np.concatenate(out)
    if not a.no_decreak:
        audio, frac = decreak(audio, sr)
        if frac: log(f"vocal-fry tremble repaired in {frac:.1%} of frames")
    audio = model.watermarker.apply_watermark(audio, sample_rate=sr)
    import soundfile as sf
    sf.write(a.out, audio, sr)
    log(f"wrote {a.out}: {len(audio) / sr:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
