#!/usr/bin/env python3
"""Production non-speech labeling of audios2 gap clips: CLAP-v3 gate -> Qwen3-Omni (vLLM, HF fallback) -> HF ledger.
Stages: candidates | prep (stream sources, cut, CLAP score, keep wav for candidates) | label | upload. All resumable."""
from __future__ import annotations
import argparse, base64, glob, json, os, random, re, subprocess, sys, threading, time, zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
SR = 48000; FRAMES, MELS = 1001, 64; MERGE = {"breathing heavy": "breathing", "breathing close": "breathing"}
CHOICES = ["kissing", "mouth sounds", "breathing", "whispering", "moaning", "normal speech", "silence", "tapping", "scratching", "crinkling", "brushing", "liquid", "other sound"]
PROMPT = ("You are labeling a short clip from an ASMR recording. Listen and pick the ONE label that best describes the MAIN sound.\n"
          "Labels: " + ", ".join(CHOICES) + ".\n"
          "Guidance: 'whispering' only if the clip is mostly whispered words; ordinary spoken words = 'normal speech'; a moan or sexual vocalization = 'moaning'; "
          "lip/kiss sounds with no words = 'kissing'; wet clicks, lip smacks, licking, tongue sounds = 'mouth sounds'; audible in/out breaths with no words = 'breathing'; near-silent = 'silence'.\n"
          "Answer with only the label.")
def parse(text):
    t = (text or "").lower().strip().strip('."\'`*:').replace("_", " ")
    for c in sorted(CHOICES, key=len, reverse=True):
        if c in t: return c
    if "breath" in t: return "breathing"
    if "kiss" in t: return "kissing"
    if "whisper" in t: return "whispering"
    if "moan" in t: return "moaning"
    if "mouth" in t or "lip" in t or "lick" in t: return "mouth sounds"
    if "speech" in t or "talk" in t or "speak" in t or "voice" in t: return "normal speech"
    if "silen" in t or "quiet" in t: return "silence"
    return None

# ---------- stage: candidates ----------
def stage_candidates(a):
    from huggingface_hub import hf_hub_download, HfApi
    out = a.work / "candidates.jsonl"
    if out.exists(): log("candidates exist"); return
    seen = set(); rows = []
    man = hf_hub_download("aoxo/clap-ft-data", "clap_finetune_manifest.jsonl", repo_type="dataset")
    for l in open(man):
        r = json.loads(l); au = r["audio"]
        if r["uid"] in seen: continue
        seen.add(r["uid"]); rows.append({"uid": r["uid"], "source": au["source"], "start": au["start"], "duration": au["duration"], "old": r["trigger"]})
    api = HfApi()
    for f in api.list_repo_files("aoxo/clap-ft-data", repo_type="dataset"):
        if f.startswith("ledgers/mac/gemini_audios2"):
            p = hf_hub_download("aoxo/clap-ft-data", f, repo_type="dataset")
            for l in open(p):
                try: r = json.loads(l)
                except Exception: continue
                if r.get("gemini_label") == "reject" and r.get("duration") and r["uid"] not in seen:
                    seen.add(r["uid"]); rows.append({"uid": r["uid"], "source": r["source"], "start": r["start"], "duration": r["duration"], "old": "reject"})
    pro = hf_hub_download("aoxo/clap-ft-data", "pilot/pilot_labels_pro.jsonl", repo_type="dataset"); pro_uids = {json.loads(l)["uid"] for l in open(pro)}
    rows = [r for r in rows if r["uid"] not in pro_uids]
    for r in rows: r["cut_start"] = round(max(0.0, r["start"] - 1.0), 3); r["cut_duration"] = round(min(8.0, r["duration"] + 2.0), 3)
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    log(f"candidates: {len(rows)} clips, {len({r['source'] for r in rows})} sources, old-label dist {Counter(r['old'] for r in rows).most_common(8)}")

# ---------- stage: prep (cut + CLAP gate) ----------
class Clap:
    def __init__(self, dev="cuda"):
        import torch; from transformers import ClapModel, ClapFeatureExtractor; from huggingface_hub import snapshot_download
        d = snapshot_download("aoxo/clap-htsat-unfused-asmr-v2", allow_patterns=["v3/student_all/*"]) + "/v3/student_all"
        self.torch = torch; self.dev = dev; self.fe = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        self.m = ClapModel.from_pretrained(d).to(dev).half().eval(); meta = json.load(open(d + "/vocal_meta.json")); self.classes = meta["classes"] + ["__bg__"]
        sd = torch.load(d + "/head.pt", map_location=dev); self.W = sd["weight"].T.float() * 10.0; self.b = sd["bias"].float()
        fb = np.asarray(self.fe.mel_filters_slaney, dtype=np.float32).T; self.fb = torch.from_numpy(fb).to(dev); self.win = torch.hann_window(1024, periodic=True, device=dev)
    def mel(self, wavs):
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(wavs).to(self.dev); spec = t.stft(x, 1024, 480, 1024, self.win, center=True, pad_mode="reflect", return_complex=True)
            mel = t.einsum("mf,bft->btm", self.fb, spec.real ** 2 + spec.imag ** 2); return (10.0 * t.log10(t.clamp(mel, min=1e-10)))[:, :FRAMES, :]
    def probs(self, wavs):
        t = self.torch
        with t.no_grad():
            x = self.mel(wavs).half().unsqueeze(1); e = self.m.get_audio_features(input_features=x, is_longer=t.zeros(len(x), 1, dtype=t.bool, device=self.dev))
            e = t.nn.functional.normalize(e.float(), dim=-1); return t.softmax(e @ self.W + self.b, 1).cpu().numpy()
def repeatpad(w):
    w = w[: SR * 10]
    if w.size < SR * 10: w = np.tile(w, SR * 10 // w.size); w = np.pad(w, (0, SR * 10 - w.size))
    return w.astype(np.float32)
def stage_prep(a):
    from huggingface_hub import hf_hub_download
    cands = [json.loads(l) for l in open(a.work / "candidates.jsonl")]; by_src = defaultdict(list)
    for r in cands: by_src[r["source"]].append(r)
    done_p = a.work / "prep_done_sources.txt"; done = set(done_p.read_text().split()) if done_p.exists() else set()
    todo = [s for s in by_src if s not in done]; log(f"prep: {len(by_src)} sources, {len(done)} done, {len(todo)} todo")
    clap = Clap(); (a.work / "wav").mkdir(exist_ok=True); idx = open(a.work / "clap_index.jsonl", "a"); done_f = open(done_p, "a"); lock = threading.Lock()
    bg_i = clap.classes.index("__bg__"); stats = Counter(); t0 = time.time()
    def one(src):
        rows = by_src[src]; local = None; out = []
        try:
            for i in range(5):
                try: local = hf_hub_download("aoxo/audios2", src, repo_type="dataset", local_dir=str(a.work / "tmp" / str(abs(hash(src)) % 64))); break
                except Exception:
                    if i == 4: raise
                    time.sleep(5 * 2 ** i)
            cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-i", local, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
            wav = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True, timeout=900).stdout, dtype=np.float32)
            for r in rows:
                s0 = int(r["cut_start"] * SR); seg = wav[s0: s0 + int(r["cut_duration"] * SR)]
                if seg.size < SR // 2: continue
                out.append((r, seg))
        except Exception as e: log(f"src fail {src[:50]}: {type(e).__name__}")
        finally:
            if local and os.path.exists(local): os.remove(local)
        return src, out
    with ThreadPoolExecutor(a.workers) as ex:
        for src, out in ex.map(one, todo):
            keep = []
            for i in range(0, len(out), 64):
                ch = out[i:i+64]; P = clap.probs(np.stack([repeatpad(seg) for _, seg in ch]))
                for (r, seg), p in zip(ch, P):
                    pbg = float(p[bg_i]); best = int(np.argmax(p[:bg_i])); stats["seen"] += 1
                    if pbg < a.bg_max:
                        fp = a.work / "wav" / (r["uid"].replace("/", "__") + ".wav")
                        seg16 = seg[::3]; import soundfile as sf; sf.write(fp, seg16, 16000, subtype="PCM_16")
                        keep.append({**{k: r[k] for k in ("uid", "source", "start", "duration", "cut_start", "cut_duration", "old")}, "clap_bg": round(pbg, 4), "clap_top": clap.classes[best], "clap_top_p": round(float(p[best]), 4)}); stats["kept"] += 1
            with lock:
                for k in keep: idx.write(json.dumps(k) + "\n")
                idx.flush(); done_f.write(src + "\n"); done_f.flush(); stats["src"] += 1
                if stats["src"] % 100 == 0:
                    el = time.time() - t0; log(f"prep {stats['src']}/{len(todo)} src, seen={stats['seen']} kept={stats['kept']} ({100*stats['kept']/max(1,stats['seen']):.0f}%) {stats['src']/el*60:.0f} src/min ETA {(len(todo)-stats['src'])/max(1e-6,stats['src']/el)/60:.0f} min")
    log(f"PREP_DONE {dict(stats)}")

# ---------- stage: label ----------
def vllm_up(port=8000):
    import urllib.request
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5); return True
    except Exception: return False
def label_vllm(a, rows, ledger, port=8000):
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    def one(r):
        b64 = base64.b64encode(open(a.work / "wav" / (r["uid"].replace("/", "__") + ".wav"), "rb").read()).decode()
        body = {"model": a.model, "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}, {"type": "text", "text": PROMPT}]}], "max_tokens": 12, "temperature": 0}
        for i in range(4):
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as resp: d = json.load(resp)
                txt = d["choices"][0]["message"]["content"]; return {"uid": r["uid"], "raw": txt.strip(), "label": parse(txt), "labeler": "qwen3-omni-30b-vllm"}
            except Exception as e:
                if i == 3: return {"uid": r["uid"], "raw": None, "label": None, "error": f"{type(e).__name__}: {str(e)[:80]}"}
                time.sleep(3)
    t0 = time.time(); n = 0
    with ThreadPoolExecutor(a.concurrency) as ex:
        for res in ex.map(one, rows):
            ledger.write(json.dumps(res) + "\n"); n += 1
            if n % 500 == 0: ledger.flush(); log(f"label {n}/{len(rows)} {n/(time.time()-t0):.2f} clips/s last={res.get('raw')!r}")
def label_hf(a, rows, ledger, bs=8):
    import torch, soundfile as sf; from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    proc = Qwen3OmniMoeProcessor.from_pretrained(a.model); proc.tokenizer.padding_side = "left"
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval(); model.disable_talker()
    t0 = time.time()
    for i in range(0, len(rows), bs):
        ch = rows[i:i+bs]
        try:
            paths = [str(a.work / "wav" / (r["uid"].replace("/", "__") + ".wav")) for r in ch]; audios = [sf.read(p, dtype="float32")[0] for p in paths]
            texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "audio", "audio": p}, {"type": "text", "text": PROMPT}]}], add_generation_prompt=True, tokenize=False) for p in paths]
            inputs = proc(text=texts, audio=audios, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad(): gen = model.generate(**inputs, return_audio=False, thinker_max_new_tokens=12, thinker_do_sample=False)
            if isinstance(gen, tuple): gen = gen[0]
            for r, txt in zip(ch, proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)): ledger.write(json.dumps({"uid": r["uid"], "raw": txt.strip(), "label": parse(txt), "labeler": "qwen3-omni-30b-hf"}) + "\n")
        except Exception as e:
            for r in ch: ledger.write(json.dumps({"uid": r["uid"], "raw": None, "label": None, "error": f"{type(e).__name__}: {str(e)[:80]}"}) + "\n")
        if (i // bs) % 50 == 0: ledger.flush(); log(f"label {i+len(ch)}/{len(rows)} {(i+len(ch))/(time.time()-t0):.2f} clips/s")
def stage_label(a):
    rows = [json.loads(l) for l in open(a.work / "clap_index.jsonl")]
    led = a.work / "labels.jsonl"; done = set()
    if led.exists():
        for l in open(led): done.add(json.loads(l)["uid"])
    rows = [r for r in rows if r["uid"] not in done]
    rows.sort(key=lambda r: -(r["clap_top_p"] * (1 - r["clap_bg"])))   # most-likely-positive first
    if a.limit: rows = rows[: a.limit]
    log(f"label: {len(done)} done, {len(rows)} todo (backend={'vllm' if vllm_up() else 'hf'})")
    with open(led, "a") as ledger:
        if vllm_up(): label_vllm(a, rows, ledger)
        else: label_hf(a, rows, ledger)
    log("LABEL_DONE")
def stage_upload(a):
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(a.work / "labels.jsonl"), path_in_repo="v2/audios2_qwen3omni_labels.jsonl", repo_id="aoxo/clap-ft-data", repo_type="dataset", commit_message="audios2 Qwen3-Omni non-speech labels (incremental)")
    if (a.work / "clap_index.jsonl").exists(): HfApi().upload_file(path_or_fileobj=str(a.work / "clap_index.jsonl"), path_in_repo="v2/audios2_clap_gate_index.jsonl", repo_id="aoxo/clap-ft-data", repo_type="dataset", commit_message="audios2 CLAP-v3 gate index")
    log("UPLOAD_DONE")
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, default=Path("/workspace/lab")); ap.add_argument("--stage", default="candidates,prep,label,upload")
    ap.add_argument("--workers", type=int, default=20); ap.add_argument("--bg-max", type=float, default=0.5); ap.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--concurrency", type=int, default=32); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True)
    for st in a.stage.split(","): {"candidates": stage_candidates, "prep": stage_prep, "label": stage_label, "upload": stage_upload}[st](a)
if __name__ == "__main__": main()
