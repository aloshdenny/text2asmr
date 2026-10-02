#!/usr/bin/env python3
"""When does each sound happen? Gemini 3.1 Pro gives time spans for labels we already trust.

Human and fused labels say *which* sounds a clip holds, not when: "moaning + whispering" could overlap or take turns,
and last 0.5 s or the whole clip. Told the labels, the model only has to place them in time, a narrower question
than open labelling. Before trusting its timestamps:

  --validate  builds clips with known answers by splicing clean single-sound clips at known positions (sequential,
              overlapping, or one sound in silence), asks for the spans, and scores onset / offset error, IoU, and
              whether overlap vs sequence is recovered
  --annotate  asks for spans on clips with established labels (people's labels, else fused P >= --min-p)

  python gemini_timing.py --validate --fused fused.jsonl --clips D:/t2a/pool D:/t2a/pool_yt --n 60 --out timing_val/
  python gemini_timing.py --annotate --fused fused.jsonl --humans humans/*.jsonl --clips ... --out timing/
"""
from __future__ import annotations
import argparse, base64, glob, io, json, os, random, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_local_labels import load

SR = 24000
MODEL = "google/gemini-3.1-pro-preview"
CLEAN = ["tapping", "crinkling", "whispering", "moaning", "breathing", "normal speech", "liquid", "scratching", "brushing",
         "oral sounds", "paper rustling", "fabric rustling"]


def prompt(dur: float, labels: list[str]) -> str:
    return (f"This audio clip is {dur:.1f} seconds long and contains these sounds: {', '.join(labels)}. For each sound, "
            "list every time span in which it can be heard, as [start, end] in seconds with one decimal. Spans of "
            "different sounds may overlap. Answer with JSON only: {\"spans\": {\"<sound>\": [[start, end], ...]}}")


def ask(mp3: bytes, text: str, key: str) -> tuple[dict, str, float]:
    body = {"model": MODEL, "temperature": 0, "max_tokens": 4000, "usage": {"include": True},
            "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                         {"type": "input_audio", "input_audio": {"data": base64.b64encode(mp3).decode(), "format": "mp3"}}]}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    err = ""
    for i in range(6):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            txt = r["choices"][0]["message"]["content"] or ""
            u = r.get("usage") or {}
            cost = float(u.get("cost") or 0) + float((u.get("cost_details") or {}).get("upstream_inference_cost") or 0)
            m = re.search(r"\{.*\}", txt, re.S)
            spans = (json.loads(m.group(0)).get("spans") if m else None) or {}
            return {k.strip().lower(): [[float(s), float(e)] for s, e in v] for k, v in spans.items()}, txt[-400:], cost
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(min(60, 5 * 2 ** i))
    return {}, f"ERROR {err}", 0.0


def to_mp3(y: np.ndarray) -> bytes:
    buf = io.BytesIO(); sf.write(buf, y, SR, format="WAV")
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "wav", "-i", "-", "-ac", "1", "-b:a", "64k", "-f", "mp3", "-"],
                       input=buf.getvalue(), capture_output=True, check=True)
    return r.stdout


def active(y: np.ndarray, hop: int = 480) -> tuple[float, float] | None:
    """First and last 20 ms frame within 25 dB of the loudest: where the sound really is."""
    n = len(y) // hop
    if n == 0: return None
    db = 20 * np.log10(np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    on = np.where(db > db.max() - 25)[0]
    return (on[0] * hop / SR, (on[-1] + 1) * hop / SR) if len(on) else None


def excerpt(y: np.ndarray, sec: float) -> np.ndarray:
    """The loudest `sec`-second stretch of a clip."""
    w = int(sec * SR)
    if len(y) <= w: return y
    e = np.convolve(y ** 2, np.ones(SR // 10), "same")
    starts = np.arange(0, len(y) - w, SR // 20)
    best = max(starts, key=lambda s: e[s:s + w].sum())
    return y[best:best + w]


def iou(pred: list[list[float]], truth: tuple[float, float], dur: float) -> float:
    t = np.zeros(int(dur * 10) + 1, bool); p = t.copy()
    t[int(truth[0] * 10):int(truth[1] * 10) + 1] = True
    for s, e in pred: p[max(0, int(s * 10)):int(e * 10) + 1] = True
    return float((t & p).sum() / max(1, (t | p).sum()))


def find_audio(dirs: list[Path]) -> dict[str, str]:
    out = {}
    for d in dirs:
        for f in glob.glob(str(d / "pool_*" / "clips" / "*.mp3")): out.setdefault(Path(f).stem, f)
    return out


def validate(a, key: str) -> int:
    F = {json.loads(l)["uid"]: json.loads(l)["probs"] for l in open(a.fused, encoding="utf-8")}
    audio = find_audio(a.clips)
    san = lambda u: re.sub(r"[^A-Za-z0-9_.-]", "_", u)
    # clean single-sound clips: one class certain, every other class near zero
    pure: dict[str, list[str]] = {c: [] for c in CLEAN}
    for u, p in F.items():
        if san(u) not in audio: continue
        top = [c for c, v in p.items() if v >= 0.95]
        if len(top) == 1 and top[0] in pure and all(v <= 0.1 for c, v in p.items() if c != top[0]): pure[top[0]].append(u)
    usable = [c for c, v in pure.items() if len(v) >= 4]
    print("clean clips per sound:", {c: len(v) for c, v in pure.items()}, flush=True)
    rng = random.Random(7)
    cases = []
    for i in range(a.n):
        layout = ["sequential", "overlap", "single"][i % 3]
        la, lb = rng.sample(usable, 2)
        ya = excerpt(load(audio[san(rng.choice(pure[la]))], SR), 2.5 if layout != "overlap" else 3.5)
        yb = excerpt(load(audio[san(rng.choice(pure[lb]))], SR), 2.5 if layout != "overlap" else 3.5)
        ya, yb = ya / (np.abs(ya).max() + 1e-9) * 0.5, yb / (np.abs(yb).max() + 1e-9) * 0.5
        if layout == "sequential": place = {la: (ya, 0.6), lb: (yb, 3.9)}; dur = 7.0
        elif layout == "overlap": place = {la: (ya, 0.3), lb: (yb, 2.4)}; dur = 6.5
        else: place = {la: (ya, rng.uniform(1.0, 3.0))}; dur = 6.0
        y = np.zeros(int(dur * SR), np.float32); truth = {}
        for lab, (seg, at) in place.items():
            s = int(at * SR); y[s:s + len(seg)] += seg[: len(y) - s]
            act = active(seg)
            if act: truth[lab] = (round(at + act[0], 2), round(at + act[1], 2))
        cases.append({"id": i, "layout": layout, "dur": dur, "truth": truth, "mp3": to_mp3(y)})

    def run(c):
        spans, raw, cost = ask(c["mp3"], prompt(c["dur"], list(c["truth"])), key)
        return {**{k: v for k, v in c.items() if k != "mp3"}, "pred": spans, "raw": raw, "cost": cost}

    with ThreadPoolExecutor(8) as ex: res = list(ex.map(run, cases))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res), encoding="utf-8")
    on, off, ious, olap = [], [], [], []
    for r in res:
        for lab, (ts, te) in r["truth"].items():
            pred = r["pred"].get(lab, [])
            if not pred: ious.append(0.0); continue
            on.append(abs(min(s for s, _ in pred) - ts)); off.append(abs(max(e for _, e in pred) - te))
            ious.append(iou(pred, (ts, te), r["dur"]))
        if r["layout"] in ("sequential", "overlap") and len(r["truth"]) == 2:
            (a1, a2), (b1, b2) = [(min(s for s, _ in r["pred"].get(l, [[0, 0]])), max(e for _, e in r["pred"].get(l, [[0, 0]]))) for l in r["truth"]]
            olap.append((min(a2, b2) - max(a1, b1) > 0.3) == (r["layout"] == "overlap"))
    q = lambda xs, f: float(np.quantile(xs, f)) if xs else float("nan")
    rep = {"n": len(res), "cost": round(sum(r["cost"] for r in res), 3), "errors": sum(r["raw"].startswith("ERROR") for r in res),
           "onset_err_median": round(q(on, 0.5), 2), "onset_err_p90": round(q(on, 0.9), 2),
           "offset_err_median": round(q(off, 0.5), 2), "offset_err_p90": round(q(off, 0.9), 2),
           "iou_mean": round(float(np.mean(ious)), 3) if ious else None,
           "overlap_vs_sequence_correct": f"{sum(olap)}/{len(olap)}",
           "by_layout": {lay: round(float(np.mean([iou(r["pred"].get(l, []), t, r["dur"]) for r in res if r["layout"] == lay
                                                     for l, t in r["truth"].items()])), 3) for lay in ("sequential", "overlap", "single")}}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--clips", type=Path, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY_GS"]
    if a.validate: return validate(a, key)
    sys.exit("only --validate is implemented until the validation says the spans can be trusted")


if __name__ == "__main__":
    raise SystemExit(main())
