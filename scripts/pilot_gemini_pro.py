#!/usr/bin/env python3
"""Pilot: relabel a stratified sample of existing gap clips with Gemini Pro, audio-only, then probe learnability."""
from __future__ import annotations
import argparse, base64, json, os, random, subprocess, threading, time, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

KEYS = ['brushing', 'tapping', 'crinkling', 'rustling', 'scratching', 'clinking', 'breathing close', 'breathing heavy', 'kissing',
        'footsteps', 'fabric rustling', 'paper rustling', 'page turning', 'liquid', 'mouth sounds', 'breathing', 'blowing',
        'hand movements', 'microphone touching', 'glass', 'wood', 'sticky', 'cutting', 'whispering']
PROJECT = "project-9da5a2fe-3df4-485e-9a9"; SR = 48000
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def prompt_text() -> str:
    keys = ", ".join(KEYS)
    return ("You are classifying a short audio clip from an ASMR audio corpus. Listen to the clip and decide what it actually contains.\n\n"
        f"Known trigger sound categories: {keys}\n\n"
        "Rules, in priority order:\n"
        "1. If the clip reasonably fits one of the known categories -- including close or approximate matches, e.g. 'wood tapping' or 'finger tapping' both belong under 'tapping', not a new label -- answer with that exact category name. Prefer an existing category over inventing a new one whenever the sound is fundamentally the same kind of thing, even if the material, intensity, or exact technique differs.\n"
        "2. If the clip is NOT a non-speech trigger sound at all (e.g. it's actually speech, moaning, silence, or something else that isn't a discrete sound effect), answer with exactly: reject\n"
        "3. Only if the clip is a real, distinct non-speech sound that genuinely isn't a variant of any known category (a different kind of sound entirely, not just a different flavor of an existing one) -- answer with a short (1-3 word), generic category name for it, not a specific description. Write it the way a new entry in the known list above would look (e.g. 'zipper' or 'page turning', not 'the sound of a metal zipper being pulled slowly'), so it can be reused as-is for other similar clips instead of every clip getting its own one-off phrasing.\n\n"
        "Respond with JSON: {\"label\": \"...\"}")

def cut_flac(src_path, start, dur, out_path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src_path),
                    "-ar", str(SR), "-ac", "1", "-sample_fmt", "s16", "-c:a", "flac", str(out_path)], check=True, timeout=120)

def gemini(model, flac_bytes, token, tries=6):
    body = {"contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "audio/flac", "data": base64.b64encode(flac_bytes).decode()}}, {"text": prompt_text()}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": {"type": "OBJECT", "properties": {"label": {"type": "STRING"}}, "required": ["label"]}}}
    url = f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/global/publishers/google/models/{model}:generateContent"
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r: d = json.load(r)
            if (d.get("promptFeedback") or {}).get("blockReason"): return "reject", f"blocked:{d['promptFeedback']['blockReason']}"
            c = d.get("candidates") or []
            if not c: return None, f"nocand:{json.dumps(d)[:120]}"
            txt = "".join(p.get("text", "") for p in c[0].get("content", {}).get("parts", []))
            lab = json.loads(txt).get("label")
            return (lab.strip().lower() if isinstance(lab, str) else None), None
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code == 401: return None, "AUTH_EXPIRED"
            if e.code in (429, 500, 503, 504) and i < tries - 1: time.sleep(min(60, 3 * 2 ** i)); continue
            return None, f"http{e.code}:{msg}"
        except Exception as e:
            if i < tries - 1: time.sleep(3 * 2 ** i); continue
            return None, f"{type(e).__name__}:{str(e)[:120]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True); ap.add_argument("--out", type=Path, default=Path("/workspace/pilot"))
    ap.add_argument("--per-class", type=int, default=120); ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--pad", type=float, default=1.0); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ledger-name", default="pilot_labels.jsonl"); ap.add_argument("--reuse-cuts", action="store_true")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); (a.out / "flac").mkdir(exist_ok=True)
    rng = random.Random(a.seed); token = os.environ["GCS_TOKEN"]; hf = os.environ["HF_TOKEN"]
    by = defaultdict(list)
    for l in open(a.subset):
        r = json.loads(l); by[r["label"]].append(r)
    sample = []
    for c, rows in by.items():
        rng.shuffle(rows); sample += rows[: a.per_class]
    rng.shuffle(sample); log(f"sample={len(sample)} classes={len(by)} sources={len({r['source'] for r in sample})}")
    ledger = a.out / a.ledger_name; done = {}
    if ledger.exists():
        for l in open(ledger): r = json.loads(l); done[r["uid"]] = r
    todo = [r for r in sample if r["uid"] not in done or done[r["uid"]].get("error") == "AUTH_EXPIRED"]
    log(f"already labeled={len(done)} todo={len(todo)}")
    by_src = defaultdict(list)
    for r in todo: by_src[r["source"]].append(r)
    from huggingface_hub import hf_hub_download
    out_f = open(ledger, "a"); lock = threading.Lock(); stats = Counter(); t0 = time.time()
    def one_source(src):
        rows = by_src[src]; local = None
        try:
            fps = {r["uid"]: a.out / "flac" / (r["uid"].replace("/", "__") + ".flac") for r in rows}
            if a.reuse_cuts and all(f.exists() for f in fps.values()): local = None
            else:
              for i in range(5):
                try: local = hf_hub_download("aoxo/audios2", src, repo_type="dataset", token=hf, local_dir=f"/workspace/tmp_src/{threading.get_ident()%97}"); break
                except Exception:
                    if i == 4: raise
                    time.sleep(5 * 2 ** i)
            for r in rows:
                s0 = max(0.0, float(r["start"]) - a.pad); d0 = float(r["duration"]) + a.pad + (float(r["start"]) - s0)
                fp = fps[r["uid"]]
                try:
                    if not (a.reuse_cuts and fp.exists()): cut_flac(local, s0, d0, fp)
                except Exception as e:
                    rec = {**{k: r[k] for k in ("uid","label","source","start","duration")}, "new_label": None, "error": f"cut:{type(e).__name__}"}
                else:
                    lab, err = gemini(a.model, fp.read_bytes(), token)
                    rec = {**{k: r[k] for k in ("uid","label","source","start","duration")}, "cut_start": round(s0,3), "cut_duration": round(d0,3), "new_label": lab, "error": err, "model": a.model}
                with lock:
                    out_f.write(json.dumps(rec) + "\n"); out_f.flush(); stats["done"] += 1
                    if rec["error"]: stats["err"] += 1
                    if rec.get("new_label") == r["label"]: stats["agree"] += 1
                    if stats["done"] % 100 == 0: log(f"{stats['done']}/{len(todo)} agree={stats['agree']} err={stats['err']} {stats['done']/(time.time()-t0)*60:.0f}/min")
                if err == "AUTH_EXPIRED": raise SystemExit("token expired")
        except SystemExit: raise
        except Exception as e:
            log(f"source fail {src}: {type(e).__name__}: {str(e)[:100]}")
        finally:
            if local:
                try: os.remove(local)
                except Exception: pass
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one_source, list(by_src)))
    log(f"LABEL_DONE {dict(stats)}")

if __name__ == "__main__": main()
