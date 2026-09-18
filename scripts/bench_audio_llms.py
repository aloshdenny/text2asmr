#!/usr/bin/env python3
"""Benchmark open audio-LLM labelers against Gemini-Pro audio-only labels on the pilot clips.
Stage cut: re-cut a stratified sample of Pro-labeled clips (16 kHz mono wav). Stage label: run each model. Stage score: agreement vs Pro + CLAP-base linear-probe learnability."""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, sys, time, zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
MERGE = {"breathing heavy": "breathing", "breathing close": "breathing"}
VOCAL = ["reject", "whispering", "kissing", "mouth sounds", "breathing"]
CHOICES = ["kissing", "mouth sounds", "breathing", "whispering", "moaning", "tapping", "scratching", "crinkling", "brushing", "page turning", "liquid", "other trigger sound", "normal speech", "silence"]
TO_VOCAL = {"moaning": "reject", "normal speech": "reject", "silence": "reject", "other trigger sound": "reject", "tapping": "reject", "scratching": "reject", "crinkling": "reject", "brushing": "reject", "page turning": "reject", "liquid": "reject"}
PROMPT = ("This is a short clip from an ASMR recording. Listen carefully. Which ONE label best describes the main sound in the clip? "
          "Choose exactly one from this list: " + ", ".join(CHOICES) + ". Answer with only the label, nothing else.")
def parse(text: str):
    t = (text or "").lower().strip().strip('."\'`*').replace("_", " ")
    for c in sorted(CHOICES, key=len, reverse=True):
        if c in t: return c
    if "breath" in t: return "breathing"
    if "kiss" in t: return "kissing"
    if "whisper" in t: return "whispering"
    if "mouth" in t or "lip" in t or "lick" in t: return "mouth sounds"
    if "speech" in t or "talk" in t or "speak" in t or "voice" in t: return "normal speech"
    if "silen" in t or "quiet" in t or "nothing" in t: return "silence"
    return None

def stage_cut(a):
    from huggingface_hub import hf_hub_download
    rows = [json.loads(l) for l in open(hf_hub_download("aoxo/clap-ft-data", "pilot/pilot_labels_pro.jsonl", repo_type="dataset"))]
    rows = {r["uid"]: r for r in rows if r.get("new_label") and not r.get("error")}.values()
    by = defaultdict(list)
    for r in rows: by[MERGE.get(r["new_label"], r["new_label"])].append(r)
    rng = random.Random(0); quota = {"reject": 500, "whispering": 500, "kissing": 400, "mouth sounds": 300, "breathing": 200}
    sample = []
    for c, q in quota.items(): rs = by[c]; rng.shuffle(rs); sample += rs[:q]
    others = [r for c, rs in by.items() if c not in quota for r in rs]; rng.shuffle(others); sample += others[:150]
    log(f"sample {len(sample)} clips from {len({r['source'] for r in sample})} sources; {Counter(MERGE.get(r['new_label'], r['new_label']) for r in sample).most_common(8)}")
    (a.work / "wav").mkdir(parents=True, exist_ok=True); by_src = defaultdict(list)
    for r in sample: by_src[r["source"]].append(r)
    def one(src):
        local = None; out = []
        try:
            for i in range(4):
                try: local = hf_hub_download("aoxo/audios2", src, repo_type="dataset", local_dir=str(a.work / "tmp" / str(abs(hash(src)) % 32))); break
                except Exception:
                    if i == 3: raise
                    time.sleep(5 * 2 ** i)
            for r in by_src[src]:
                fp = a.work / "wav" / (r["uid"].replace("/", "__") + ".wav")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{r['cut_start']:.3f}", "-t", f"{r['cut_duration']:.3f}", "-i", local, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(fp)], check=True, timeout=120)
                out.append({"uid": r["uid"], "pro": r["new_label"], "pro_vocal": MERGE.get(r["new_label"], r["new_label"]) if MERGE.get(r["new_label"], r["new_label"]) in VOCAL else "reject", "wav": str(fp), "source": src})
        except Exception as e: log(f"fail {src[:60]}: {type(e).__name__}")
        finally:
            if local and os.path.exists(local): os.remove(local)
        return out
    clips = []
    with ThreadPoolExecutor(16) as ex:
        for out in ex.map(one, list(by_src)): clips += out
    json.dump(clips, open(a.work / "clips.json", "w")); log(f"CUT_DONE {len(clips)} clips")

def load_audio(path, sr=16000):
    import soundfile as sf
    x, s = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    return x, s

def run_qwen2_audio(a, clips):
    import torch; from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    mid = "Qwen/Qwen2-Audio-7B-Instruct"; proc = AutoProcessor.from_pretrained(mid); model = Qwen2AudioForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    out = []
    for i, c in enumerate(clips):
        try:
            audio, sr = load_audio(c["wav"]); conv = [{"role": "user", "content": [{"type": "audio", "audio_url": c["wav"]}, {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            inputs = proc(text=text, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad(): gen = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            resp = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            out.append({"uid": c["uid"], "raw": resp.strip(), "label": parse(resp)})
        except Exception as e: out.append({"uid": c["uid"], "raw": None, "label": None, "error": f"{type(e).__name__}: {str(e)[:100]}"})
        if i % 200 == 0: log(f"qwen2-audio {i}/{len(clips)} last={out[-1].get('raw')!r}")
    del model; torch.cuda.empty_cache(); return out

def run_voxtral(a, clips):
    import torch; from transformers import VoxtralForConditionalGeneration, AutoProcessor
    mid = "mistralai/Voxtral-Mini-3B-2507"; proc = AutoProcessor.from_pretrained(mid); model = VoxtralForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    out = []
    for i, c in enumerate(clips):
        try:
            conv = [{"role": "user", "content": [{"type": "audio", "path": c["wav"]}, {"type": "text", "text": PROMPT}]}]
            inputs = proc.apply_chat_template(conv, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
            with torch.no_grad(): gen = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            resp = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            out.append({"uid": c["uid"], "raw": resp.strip(), "label": parse(resp)})
        except Exception as e: out.append({"uid": c["uid"], "raw": None, "label": None, "error": f"{type(e).__name__}: {str(e)[:100]}"})
        if i % 200 == 0: log(f"voxtral {i}/{len(clips)} last={out[-1].get('raw')!r}")
    del model; torch.cuda.empty_cache(); return out

def run_qwen25_omni(a, clips):
    import torch; from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    mid = "Qwen/Qwen2.5-Omni-7B"; proc = Qwen2_5OmniProcessor.from_pretrained(mid)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda", enable_audio_output=False).eval()
    out = []
    for i, c in enumerate(clips):
        try:
            audio, sr = load_audio(c["wav"]); conv = [{"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
                                                {"role": "user", "content": [{"type": "audio", "audio": c["wav"]}, {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            inputs = proc(text=text, audio=[audio], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad(): gen = model.generate(**inputs, max_new_tokens=16, do_sample=False, return_audio=False)
            if isinstance(gen, tuple): gen = gen[0]
            resp = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            out.append({"uid": c["uid"], "raw": resp.strip(), "label": parse(resp)})
        except Exception as e: out.append({"uid": c["uid"], "raw": None, "label": None, "error": f"{type(e).__name__}: {str(e)[:100]}"})
        if i % 200 == 0: log(f"qwen2.5-omni {i}/{len(clips)} last={out[-1].get('raw')!r}")
    del model; torch.cuda.empty_cache(); return out

MODELS = {"qwen2-audio-7b": run_qwen2_audio, "voxtral-mini-3b": run_voxtral, "qwen2.5-omni-7b": run_qwen25_omni}

def stage_label(a):
    clips = json.load(open(a.work / "clips.json"))
    for name in a.models.split(","):
        outp = a.work / f"labels_{name}.jsonl"
        if outp.exists(): log(f"{name}: exists, skipping"); continue
        log(f"=== {name}"); t0 = time.time()
        try: res = MODELS[name](a, clips)
        except Exception as e: log(f"{name} FAILED to load/run: {type(e).__name__}: {str(e)[:200]}"); continue
        with open(outp, "w") as f:
            for r in res: f.write(json.dumps(r) + "\n")
        ok = [r for r in res if r["label"]]; log(f"{name}: {len(ok)}/{len(res)} parsed in {(time.time()-t0)/60:.1f} min; raw sample: {[r['raw'] for r in res[:5]]}")

def stage_score(a):
    import torch; from transformers import ClapModel, ClapFeatureExtractor
    clips = json.load(open(a.work / "clips.json")); pro = {c["uid"]: c for c in clips}
    fe = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused"); clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").cuda().eval()
    embp = a.work / "clap_emb.npz"
    if embp.exists(): z = np.load(embp); E = {u: e for u, e in zip(z["uids"], z["emb"])}
    else:
        E = {}
        for i in range(0, len(clips), 32):
            ch = clips[i:i+32]; wavs = []
            for c in ch:
                out = subprocess.run(["ffmpeg", "-v", "error", "-i", c["wav"], "-f", "f32le", "-ac", "1", "-ar", "48000", "-"], capture_output=True, check=True).stdout
                w = np.frombuffer(out, dtype=np.float32)[:480000]; wavs.append(w)
            feats = np.stack([fe(w, sampling_rate=48000, return_tensors="np")["input_features"][0].reshape(1001, 64) for w in wavs])
            with torch.no_grad(): e = clap.audio_model(input_features=torch.from_numpy(feats).unsqueeze(1).cuda(), is_longer=torch.zeros(len(ch), 1, dtype=torch.bool, device="cuda")).pooler_output.float().cpu().numpy()
            for c, v in zip(ch, e): E[c["uid"]] = v
        np.savez(embp, uids=np.array(list(E)), emb=np.stack(list(E.values())))
    rng = random.Random(0); dev = "cuda"
    def probe(labels_by_uid, classes, cap=100, reps=5):
        accs = []
        for _ in range(reps):
            tr, ev = [], []
            for i, c in enumerate(classes):
                us = [u for u, l in labels_by_uid.items() if l == c and u in E]; rng.shuffle(us); us = us[:cap]; cut = int(len(us) * 0.75); tr += [(u, i) for u in us[:cut]]; ev += [(u, i) for u in us[cut:]]
            if len(ev) < 12 or len({y for _, y in ev}) < len(classes): return None
            Xtr = torch.tensor(np.stack([E[u] for u, _ in tr])).to(dev); Xev = torch.tensor(np.stack([E[u] for u, _ in ev])).to(dev); ytr = torch.tensor([y for _, y in tr], device=dev); yev = torch.tensor([y for _, y in ev], device=dev)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xev = (Xev - mu) / sd
            W = torch.zeros(Xtr.shape[1], len(classes), device=dev, requires_grad=True); b = torch.zeros(len(classes), device=dev, requires_grad=True); opt = torch.optim.Adam([W, b], lr=1e-2)
            for _ in range(500):
                loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr) + 1e-3 * (W ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): accs.append(((Xev @ W + b).argmax(1) == yev).float().mean().item())
        return round(float(np.mean(accs)), 3)
    report = {}
    labelsets = {"gemini-pro": {c["uid"]: c["pro_vocal"] for c in clips}}
    for name in a.models.split(","):
        p = a.work / f"labels_{name}.jsonl"
        if not p.exists(): continue
        rows = [json.loads(l) for l in open(p)]; m = {r["uid"]: (TO_VOCAL.get(r["label"], r["label"]) if r["label"] else None) for r in rows}
        labelsets[name] = {u: l for u, l in m.items() if l}
        n = len(rows); parsed = sum(1 for r in rows if r["label"]); agree = sum(1 for u, l in m.items() if l and l == pro[u]["pro_vocal"])
        per = defaultdict(lambda: [0, 0]); conf = defaultdict(Counter)
        for u, l in m.items():
            g = pro[u]["pro_vocal"]; per[g][1] += 1; conf[g][l] += 1
            if l == g: per[g][0] += 1
        report[name] = {"parsed": f"{parsed}/{n}", "agree_vocal5": round(agree / max(1, parsed), 3), "recall_vs_pro": {g: round(v[0] / v[1], 2) for g, v in per.items()}, "confusion": {g: conf[g].most_common(3) for g in per},
                        "raw_dist": Counter(r["label"] for r in rows).most_common(8)}
    for name, lab in labelsets.items():
        report.setdefault(name, {})["probe"] = {"4way reject/whisper/kiss/mouth": probe(lab, ["reject", "whispering", "kissing", "mouth sounds"]), "reject vs whispering": probe(lab, ["reject", "whispering"]),
                                                "whispering vs kissing": probe(lab, ["whispering", "kissing"]), "reject vs kissing": probe(lab, ["reject", "kissing"]), "whispering vs mouth sounds": probe(lab, ["whispering", "mouth sounds"]), "kissing vs breathing": probe(lab, ["kissing", "breathing"])}
    json.dump(report, open(a.work / "report.json", "w"), indent=1, default=str)
    for name, r in report.items(): log(f"{name}: " + json.dumps(r, default=str)[:900])
    log("SCORE_DONE")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, default=Path("/workspace/bench")); ap.add_argument("--stage", default="cut,label,score"); ap.add_argument("--models", default="qwen2-audio-7b,voxtral-mini-3b,qwen2.5-omni-7b")
    a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True)
    if "cut" in a.stage and not (a.work / "clips.json").exists(): stage_cut(a)
    if "label" in a.stage: stage_label(a)
    if "score" in a.stage: stage_score(a)
if __name__ == "__main__": main()
