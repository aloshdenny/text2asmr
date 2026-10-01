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
import argparse, base64, json, os, re, threading, time, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LABELS = ["breathing", "kissing", "oral sounds", "moaning", "whispering", "normal speech", "tapping", "scratching",
          "crinkling", "brushing", "liquid", "microphone touching", "sticky", "fabric rustling", "paper rustling",
          "cutting", "background music", "silence / room tone", "something else"]
SYN = {"silence": "silence / room tone", "room tone": "silence / room tone", "mouth sounds": "oral sounds",
       "speech": "normal speech", "talking": "normal speech", "music": "background music", "kiss": "kissing",
       "page turning": "paper rustling", "mic touching": "microphone touching", "other": "something else"}
PROMPT = ("Listen to this short ASMR audio clip. Which of these sounds can be heard in it? Choose every label that "
          "applies, only from this list: " + "; ".join(LABELS) + ". Answer with JSON only: {\"labels\": [...]}")
MODELS = {"gemini-3.1-pro": ("google/gemini-3.1-pro-preview", 12), "voxtral-small-24b": ("mistralai/voxtral-small-24b-2507", 8),
          "qwen3.8-omni-flash": ("qwen/qwen3.8-omni-flash", 8), "mimo-v2.6-flash": ("xiaomi/mimo-v2.6-flash", 8),
          "nemotron-3-nano-omni": ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", 3)}


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def norm_labels(xs) -> list[str]:
    out = []
    for x in xs or []:
        x = str(x).strip().lower(); x = SYN.get(x, x)
        if x in LABELS and x not in out: out.append(x)
    return out


def ask(model: str, mp3: bytes, key: str) -> tuple[list[str], str, float]:
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
