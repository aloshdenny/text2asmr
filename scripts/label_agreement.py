#!/usr/bin/env python3
"""How well do machine labellers agree with a human ear? Scores audio LLMs (and CLAP) against human labels.

Input is a returned ASMR Ear Check kit: the clips are read straight out of the HTML the labeller saved, so every
model hears exactly what the person heard, with the same label menu. Each model is asked for every label it
hears (ASMR clips are multi-label); CLAP gives its single top class. Writes <out>/<model>.jsonl per labeller.

  python label_agreement.py --kit kit_adi.html --labels adi_150.jsonl --out D:\\t2a\\labelcmp
  python label_agreement.py --kit kit_adi.html --labels adi_150.jsonl --out out --clap D:\\t2a\\labelcmp\\ckpt

Scoring (per model, against the human): any-overlap rate, exact-set rate, mean Jaccard, and per class precision
(model says X -> human heard X) and recall (human heard X -> model says X).
"""
from __future__ import annotations
import argparse, base64, json, os, re, threading, time, urllib.error, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LABELS = ["breathing", "kissing", "oral sounds", "moaning", "whispering", "normal speech", "tapping", "scratching",
          "crinkling", "brushing", "liquid", "spraying", "microphone touching", "sticky", "fabric rustling", "paper rustling",
          "cutting", "background music", "silence / room tone", "something else"]
SYN = {"silence": "silence / room tone", "room tone": "silence / room tone", "mouth sounds": "oral sounds",
       "speech": "normal speech", "talking": "normal speech", "music": "background music", "kiss": "kissing",
       "page turning": "paper rustling", "spray": "spraying", "spritzing": "spraying", "mic touching": "microphone touching", "other": "something else"}
PROMPT = ("Listen to this short ASMR audio clip. Which of these sounds can be heard in it? Choose every label that "
          "applies, only from this list: " + "; ".join(LABELS) + ". Answer with JSON only: {\"labels\": [...]}")
MODELS = {"gemini-3.1-pro": ("google/gemini-3.1-pro-preview", 12), "voxtral-small-24b": ("mistralai/voxtral-small-24b-2507", 8),
          "qwen3.8-omni-flash": ("qwen/qwen3.8-omni-flash", 8), "mimo-v2.6-flash": ("xiaomi/mimo-v2.6-flash", 8),
          "nemotron-3-nano-omni": ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", 3),
          "gemini-2.5-pro": ("google/gemini-2.5-pro", 8), "mimo-v2.6-pro": ("xiaomi/mimo-v2.6-pro", 4),
          "gemini-3.8-flash": ("google/gemini-3.8-flash", 8), "gpt-audio": ("openai/gpt-audio", 8),
          "gpt-audio-1.5": ("openai/gpt-audio-1.5", 8)}


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def norm_labels(xs) -> list[str]:
    out = []
    for x in xs or []:
        x = str(x).strip().lower(); x = SYN.get(x, x)
        if x in LABELS and x not in out: out.append(x)
    return out


# USD per million tokens (input, output incl. thinking) for Gemini called on Google's API directly
GOOGLE_PRICES = {"gemini-3.1-pro-preview": (2.0, 12.0), "gemini-2.5-pro": (1.25, 10.0), "gemini-3.8-flash": (0.75, 4.5)}


def ask_google(model: str, mp3: bytes) -> tuple[list[str], str, float]:
    """The same question on Google's Gemini API (GOOGLE_API_KEY, else GEMINI_API_KEY), for when the OpenRouter key
    cannot spend (OpenRouter only ever passed Gemini through to a Google key; Google bills either way)."""
    name = model.split("/", 1)[1]
    body = {"contents": [{"role": "user", "parts": [{"text": PROMPT},
                                                    {"inline_data": {"mime_type": "audio/mp3", "data": base64.b64encode(mp3).decode()}}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 8000}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent",
                                 data=json.dumps(body).encode(),
                                 headers={"x-goog-api-key": os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"],
                                          "Content-Type": "application/json"})

    def go():
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        txt = "".join(p.get("text", "") for p in ((r.get("candidates") or [{}])[0].get("content") or {}).get("parts", [])
                      if not p.get("thought"))
        u = r.get("usageMetadata") or {}
        pin, pout = GOOGLE_PRICES.get(name, (2.0, 12.0))
        cost = (u.get("promptTokenCount", 0) * pin + (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)) * pout) / 1e6
        return _parse(txt), txt[-300:], cost
    return _retry(go, "google")


# USD per million tokens, estimates where the provider does not return a cost: (text in, audio/image in, out)
OPENAI_PRICES = {"gpt-audio": (2.5, 32.0, 10.0), "gpt-audio-1.5": (2.5, 32.0, 10.0), "gpt-audio-mini": (0.6, 10.0, 2.4)}
SPECTRO_PROMPT = ("This image is a log-mel spectrogram of a 6-second ASMR audio clip: time runs left to right (0-6 s), "
                  "frequency bottom to top (Hz on the axis), brighter means louder. Judging from the spectrogram, which of "
                  "these sounds can be heard in the clip? Choose every label that applies, only from this list: "
                  + "; ".join(LABELS) + ". Answer with JSON only: {\"labels\": [...]}")


def _parse(txt: str) -> list[str]:
    m = re.search(r"\{.*\}", txt or "", re.S)
    try: return norm_labels(json.loads(m.group(0)).get("labels") if m else [])
    except Exception: return []


# Requests per minute per provider, shared by every thread in the process (override with T2A_RPM_<PROVIDER>).
# Labelling is bound by these limits, not by CPU: more workers than the limit only buys 429s. OpenAI's tier reports
# its own limit in x-ratelimit-limit-requests (400/min on the current key); Gemini's free/low tiers cap requests per day.
RPM_DEFAULT = {"openai": 350, "google": 60, "anthropic": 50, "xai": 60}
_PACE, _DEAD, _PLOCK = {}, {}, threading.Lock()


def _pace(provider: str) -> None:
    """Space requests to the provider's per-minute limit: each caller takes the next free slot, then sleeps until it."""
    gap = 60.0 / float(os.environ.get(f"T2A_RPM_{provider.upper()}", RPM_DEFAULT.get(provider, 60)))
    with _PLOCK:
        now = time.time(); slot = max(now, _PACE.get(provider, 0.0)); _PACE[provider] = slot + gap
    if slot > now: time.sleep(slot - now)


def _retry(fn, provider: str) -> tuple[list[str], str, float]:
    """Call fn at the provider's pace. A rate-limit 429 waits (Retry-After when given) and retries; an exhausted quota
    or credit balance marks the provider dead for the rest of the run, so every later clip fails fast as ERROR QUOTA
    (label_pool retries ERROR rows on its next run) instead of spending six backoffs each."""
    if provider in _DEAD: return [], f"ERROR QUOTA {_DEAD[provider]}", 0.0
    err = ""
    for i in range(6):
        _pace(provider)
        try: return fn()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:800]
            if e.code == 429 and any(k in body for k in ("insufficient_quota", "credit_balance", "PerDay", "per_day")):
                with _PLOCK:
                    if provider not in _DEAD:
                        _DEAD[provider] = re.sub(r"\s+", " ", body)[:140]
                        log(f"{provider}: quota/credit exhausted -- skipping it for the rest of this run ({_DEAD[provider][:90]})")
                return [], f"ERROR QUOTA {_DEAD[provider]}", 0.0
            flat = re.sub(r"\s+", " ", body)
            err = f"HTTP {e.code}: {flat[:140]}"
            ra = (e.headers or {}).get("retry-after")
            try: time.sleep(min(120.0, float(ra)))
            except (TypeError, ValueError): time.sleep(min(60, 5 * 2 ** i))
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:160]}"
            time.sleep(min(60, 5 * 2 ** i))
    return [], f"ERROR {err}", 0.0


def ask_openai(model: str, mp3: bytes) -> tuple[list[str], str, float]:
    """OpenAI's audio models on OpenAI's own API (OPENAI_API_KEY): the clip as input_audio, text answer only."""
    name = model.split("/", 1)[1]
    body = {"model": name, "modalities": ["text"], "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT}, {"type": "input_audio", "input_audio": {"data": base64.b64encode(mp3).decode(), "format": "mp3"}}]}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})

    def go():
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        txt = r["choices"][0]["message"].get("content") or ""
        u = r.get("usage") or {}; d = u.get("prompt_tokens_details") or {}
        tin, ain, out = OPENAI_PRICES.get(name, (2.5, 32.0, 10.0))
        audio = d.get("audio_tokens", 0)
        cost = ((u.get("prompt_tokens", 0) - audio) * tin + audio * ain + u.get("completion_tokens", 0) * out) / 1e6
        return _parse(txt), txt[-300:], cost
    return _retry(go, "openai")


def spectro_png(mp3: bytes) -> bytes:
    """A labelled log-mel spectrogram of the clip, for judges that see images but cannot hear (Claude, Grok)."""
    import io, subprocess
    import numpy as np, librosa, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "f32le", "-ac", "1", "-ar", "24000", "-"],
                         input=mp3, capture_output=True, check=True).stdout
    y = np.frombuffer(pcm, np.float32)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=24000, n_fft=1024, hop_length=240, n_mels=128), ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    librosa.display.specshow(S, sr=24000, hop_length=240, x_axis="time", y_axis="mel", ax=ax, cmap="magma", vmin=-80, vmax=0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("Hz"); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()


def ask_claude_spectro(model: str, mp3: bytes) -> tuple[list[str], str, float]:
    """Claude on the Anthropic API (ANTHROPIC_API_KEY) with the clip's spectrogram as an image: Claude cannot take audio."""
    import anthropic
    import librosa.display  # noqa: F401  (specshow)
    name = model.split("/", 1)[1]
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=4)
    img = base64.standard_b64encode(spectro_png(mp3)).decode()

    def go():
        r = client.beta.messages.create(
            model=name, max_tokens=4000, betas=["server-side-fallback-2026-06-01"], fallbacks=[{"model": "claude-opus-4-8"}],
            messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
                                                   {"type": "text", "text": SPECTRO_PROMPT}]}])
        txt = "".join(b.text for b in r.content if b.type == "text")
        if r.stop_reason == "refusal": return [], f"REFUSED {getattr(r.stop_details, 'category', '')}", 0.0
        cost = (r.usage.input_tokens * 4.0 + r.usage.output_tokens * 20.0) / 1e6
        return _parse(txt), txt[-300:], cost
    return _retry(go, "anthropic")


def ask_xai_spectro(model: str, mp3: bytes) -> tuple[list[str], str, float]:
    """Grok on xAI's API (XAI_API_KEY) with the clip's spectrogram as an image: no Grok model takes audio."""
    import librosa.display  # noqa: F401
    name = model.split("/", 1)[1]
    url = "data:image/png;base64," + base64.b64encode(spectro_png(mp3)).decode()
    body = {"model": name, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": SPECTRO_PROMPT}]}]}
    req = urllib.request.Request("https://api.x.ai/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + os.environ["XAI_API_KEY"], "Content-Type": "application/json"})

    def go():
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        txt = r["choices"][0]["message"].get("content") or ""
        u = r.get("usage") or {}
        cost = (u.get("prompt_tokens", 0) * 3.0 + u.get("completion_tokens", 0) * 15.0) / 1e6   # estimate
        return _parse(txt), txt[-300:], cost
    return _retry(go, "xai")


def ask(model: str, mp3: bytes, key: str) -> tuple[list[str], str, float]:
    """Route a judge: spectro:anthropic/... and spectro:xai/... see a spectrogram; openai/... and google/... go to the
    provider's own API when T2A_OPENAI_DIRECT / T2A_GEMINI_DIRECT is 1; everything else through OpenRouter."""
    if model.startswith("spectro:anthropic/"): return ask_claude_spectro(model.split(":", 1)[1], mp3)
    if model.startswith("spectro:xai/"): return ask_xai_spectro(model.split(":", 1)[1], mp3)
    if model.startswith("openai/") and os.environ.get("T2A_OPENAI_DIRECT") == "1": return ask_openai(model, mp3)
    if model.startswith("google/") and os.environ.get("T2A_GEMINI_DIRECT") == "1": return ask_google(model, mp3)
    body = {"model": model, "temperature": 0, "max_tokens": 4000, "usage": {"include": True},
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                         {"type": "input_audio", "input_audio": {"data": base64.b64encode(mp3).decode(), "format": "mp3"}}]}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for i in range(6):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            txt = r["choices"][0]["message"]["content"] or ""
            u = r.get("usage") or {}
            cost = float(u.get("cost") or 0) + float((u.get("cost_details") or {}).get("upstream_inference_cost") or 0)
            m = re.search(r"\{.*\}", txt, re.S)
            return norm_labels(json.loads(m.group(0)).get("labels") if m else []), txt[-300:], cost
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(min(60, 5 * 2 ** i))
    return [], f"ERROR {err}", 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", type=Path, required=True, help="the labeller's returned kit HTML (clips embedded)")
    ap.add_argument("--labels", type=Path, required=True, help="read_label_html.py output for that kit")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--clap", type=Path, default=None, help="CLAP checkpoint dir (score_pool_clap format) to score too")
    ap.add_argument("--score-only", action="store_true", help="just score what is already in --out")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.score_only: return score(a)
    html = a.kit.read_text(encoding="utf-8")
    clips = {cid: base64.b64decode(b) for cid, b in
             re.findall(r'\{"id": "(c\d+)", "dur": [\d.]+, "src": "data:audio/mpeg;base64,([A-Za-z0-9+/=]+)"\}', html)}
    human = {json.loads(l)["clip"]: json.loads(l) for l in open(a.labels, encoding="utf-8")}
    ids = sorted(set(clips) & set(human))
    log(f"{len(ids)} clips with human labels")
    key = os.environ["OPENROUTER_API_KEY_GS"]

    for name in a.models.split(","):
        model, conc = MODELS[name]
        dst = a.out / f"{name}.jsonl"
        done = {json.loads(l)["clip"] for l in open(dst, encoding="utf-8")} if dst.exists() else set()
        todo = [c for c in ids if c not in done]
        lock, spent = threading.Lock(), [0.0]

        def one(cid):
            labels, raw, cost = ask(model, clips[cid], key)
            with lock:
                spent[0] += cost
                with open(dst, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"clip": cid, "model": model, "labels": labels, "raw": raw, "cost": cost}) + "\n")
        log(f"{name}: {len(todo)} to ask ({len(done)} done)")
        with ThreadPoolExecutor(conc) as ex: list(ex.map(one, todo))
        log(f"{name}: done, ${spent[0]:.3f}")

    if a.clap:
        import io, numpy as np, soundfile as sf, librosa, sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from score_pool_clap import Scorer, repeatpad
        scorer = Scorer(a.clap); classes = scorer.classes + ["__bg__"]
        rows = []
        for cid in ids:
            try: w, sr = sf.read(io.BytesIO(clips[cid]), dtype="float32")
            except Exception: w, sr = librosa.load(io.BytesIO(clips[cid]), sr=None, mono=True)
            if w.ndim > 1: w = w.mean(1)
            w = librosa.resample(w, orig_sr=sr, target_sr=48000)
            p = scorer(repeatpad(w)[None])[0]
            top = classes[int(p.argmax())]
            rows.append({"clip": cid, "model": "clap-v7", "labels": norm_labels([top]) if top != "__bg__" else [],
                         "top": top, "p": round(float(p.max()), 3)})
        (a.out / "clap-v7.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        log(f"clap-v7: scored {len(rows)} (classes {scorer.classes})")
    log("AGREEMENT_DONE")
    return score(a)


def score(a) -> int:
    """Each labeller against the human: the pipeline's own label (the kit's hidden stratum) and every model file."""
    human = {json.loads(l)["clip"]: json.loads(l) for l in open(a.labels, encoding="utf-8")}
    H = {c: set(r["labels"]) for c, r in human.items()}
    preds = {"pipeline label": {c: set(norm_labels([{"silence": "silence / room tone"}.get(r.get("stratum"), r.get("stratum"))]))
                                for c, r in human.items() if r.get("stratum") not in (None, "uniform")}}
    for f in sorted(a.out.glob("*.jsonl")):
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        preds[f.stem] = {r["clip"]: set(r["labels"]) for r in rows if not str(r.get("raw", "")).startswith("ERROR")}
    lines = [f"{'labeller':22} {'n':>4} {'any-overlap':>11} {'exact':>6} {'jaccard':>8} {'prec':>5} {'recall':>6} {'F1':>5}"]
    per_class = {}
    for name, P in preds.items():
        cs = [c for c in P if c in H]
        if not cs: continue
        inter = sum(len(P[c] & H[c]) for c in cs); np_ = sum(len(P[c]) for c in cs); nh = sum(len(H[c]) for c in cs)
        pr, rc = inter / max(np_, 1), inter / max(nh, 1)
        jac = sum(len(P[c] & H[c]) / max(len(P[c] | H[c]), 1) for c in cs) / len(cs)
        lines.append(f"{name:22} {len(cs):4d} {sum(bool(P[c] & H[c]) for c in cs) / len(cs):11.0%} "
                     f"{sum(P[c] == H[c] for c in cs) / len(cs):6.0%} {jac:8.2f} {pr:5.0%} {rc:6.0%} {2 * pr * rc / max(pr + rc, 1e-9):5.2f}")
        pc = {}
        for lab in LABELS:
            sup = sum(lab in H[c] for c in cs); said = sum(lab in P[c] for c in cs); hit = sum(lab in P[c] and lab in H[c] for c in cs)
            if sup >= 5 or said >= 5: pc[lab] = (hit, said, sup)
        per_class[name] = pc
    support = Counter(l for c in H for l in H[c])
    lines.append("\nper class -- precision (model says X, human heard X) / recall (human heard X, model says X):")
    names = list(per_class)
    lines.append(f"{'class':20} {'human n':>7} " + " ".join(f"{n[:14]:>14}" for n in names))
    for lab, n in support.most_common():
        if n < 5: continue
        cells = []
        for nm in names:
            hit, said, sup = per_class[nm].get(lab, (0, 0, 0))
            cells.append(f"{(f'{hit / said:.0%}' if said else '-'):>5}/{(f'{hit / sup:.0%}' if sup else '-'):>4}".rjust(14))
        lines.append(f"{lab:20} {n:7d} " + " ".join(cells))
    rep = "\n".join(lines); print(rep)
    (a.out / "agreement.txt").write_text(rep + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
