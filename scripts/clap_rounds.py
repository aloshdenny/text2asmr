#!/usr/bin/env python3
"""Iterative loop: +N Gemini-Pro labels -> mel -> balanced vocal set -> train CLAP -> eval vs Pro on fixed held-out sources -> push."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, zlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/t2a/scripts")
from prep_clap_v2 import GpuMel, repeatpad, decode, ShardWriter, FRAMES, MELS, SR, MAX_S
MERGE = {"breathing heavy": "breathing", "breathing close": "breathing"}
TEXTS = {
 "reject": ["a person talking, normal speech, not an ASMR trigger sound", "spoken voice, conversation, no trigger", "silence or room tone, no ASMR trigger", "speech or moaning, not a close-mic trigger sound"],
 "whispering": ["ASMR whispering, close-mic binaural, no speech", "soft whispered voice close to the microphone", "gentle whispering", "The sound of whispering"],
 "kissing": ["ASMR kissing, close-mic binaural, no speech", "soft kisses close to a microphone", "lip kissing sounds", "The sound of kissing"],
 "mouth sounds": ["ASMR mouth sounds, close-mic binaural, no speech", "soft mouth sounds", "lip smacking close to a microphone", "The sound of mouth sounds"],
 "breathing": ["ASMR breathing, close-mic binaural, no speech", "slow deep breathing", "soft breath close to a microphone", "heavy breathing", "The sound of breathing"],
}
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def sh(cmd, **kw): return subprocess.run(cmd, shell=True, check=True, **kw)

def label_more(a, per_class):
    log(f"labeling: per-class target {per_class}")
    sh(f"cd /workspace/t2a && python scripts/pilot_gemini_pro.py --subset label_tool/subset.jsonl --out {a.pilot} --per-class {per_class} --workers {a.workers} --model {a.model} >> /workspace/label_rounds.log 2>&1")

def load_ledger(a):
    rows = {}
    for l in open(a.pilot / "pilot_labels.jsonl"):
        r = json.loads(l)
        if r.get("new_label") and not r.get("error"): rows[r["uid"]] = r
    return rows

def ensure_mel(a, ledger, gm):
    mel = a.mel; mel.mkdir(exist_ok=True); idx = mel / "index.jsonl"
    have = {json.loads(l)["uid"] for l in open(idx)} if idx.exists() else set()
    todo = [r for r in ledger.values() if r["uid"] not in have]
    log(f"mel: have={len(have)} todo={len(todo)}")
    if not todo: return
    w = ShardWriter(mel, 4000); from huggingface_hub import hf_hub_download
    from concurrent.futures import ThreadPoolExecutor
    by_src = defaultdict(list)
    for r in todo: by_src[r["source"]].append(r)
    def one(src):
        rs = by_src[src]; wavs, metas, local = [], [], None
        for r in rs:
            fp = a.pilot / "flac" / (r["uid"].replace("/", "__") + ".flac")
            try:
                if not fp.exists():
                    if local is None:
                        for i in range(5):
                            try: local = hf_hub_download("aoxo/audios2", src, repo_type="dataset", token=os.environ["HF_TOKEN"], local_dir=f"/workspace/tmp_src/{abs(hash(src)) % 64}"); break
                            except Exception:
                                if i == 4: raise
                                time.sleep(5 * 2 ** i)
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{r['cut_start']:.3f}", "-t", f"{r['cut_duration']:.3f}", "-i", local, "-ar", str(SR), "-ac", "1", "-sample_fmt", "s16", "-c:a", "flac", str(fp)], check=True, timeout=120)
                wv = decode(fp, 0.0, 12.0)
                if wv.size < SR // 4: continue
                wavs.append(repeatpad(wv)); metas.append({"uid": r["uid"], "label": r["new_label"], "text": [], "split": "", "source": src})
            except Exception as e: log(f"mel fail {r['uid'][:60]}: {type(e).__name__}")
        if local:
            try: os.remove(local)
            except Exception: pass
        return wavs, metas
    n = 0
    with ThreadPoolExecutor(a.mel_workers) as ex:
        for wavs, metas in ex.map(one, list(by_src)):
            if wavs: w.write(gm(np.stack(wavs)), metas); n += len(wavs)
            if n and n % 500 < len(wavs): log(f"mel progress {n}/{len(todo)}")
    log(f"mel done +{n}")
    import torch; torch.cuda.empty_cache()

def build_round_data(a, rnd):
    rows = [json.loads(l) for l in open(a.mel / "index.jsonl")]
    for r in rows: r["label"] = MERGE.get(r["label"], r["label"])
    cnt = Counter(r["label"] for r in rows); keep = {c for c, n in cnt.items() if n >= a.min_class and c in TEXTS}
    rows = [r for r in rows if r["label"] in keep]
    for r in rows:
        r["split"] = "eval" if zlib.crc32(r["source"].encode()) % 100 < a.eval_pct else "train"; r["text"] = TEXTS[r["label"]]
    tr = [r for r in rows if r["split"] == "train"]; ev = [r for r in rows if r["split"] == "eval"]
    ctr = Counter(r["label"] for r in tr); cap = max(a.min_class, int(a.head_frac * len(tr)))
    per = Counter(); tr2 = []
    import random; random.Random(rnd).shuffle(tr)
    for r in tr:
        if per[r["label"]] >= cap: continue
        per[r["label"]] += 1; tr2.append(r)
    d = a.out / f"round{rnd:02d}" / "data"; d.mkdir(parents=True, exist_ok=True)
    for shp in a.mel.glob("shard_*.f16"):
        dst = d / shp.name
        if not dst.exists(): dst.symlink_to(shp)
    with open(d / "index.jsonl", "w") as f:
        for r in tr2 + ev: f.write(json.dumps(r) + "\n")
    log(f"round {rnd}: labeled={len(rows)} train={len(tr2)} {dict(Counter(r['label'] for r in tr2))} eval={len(ev)} {dict(Counter(r['label'] for r in ev))}")
    return d, len(tr2), len(ev)

def train_eval(a, rnd, d, n_tr):
    out = a.out / f"round{rnd:02d}" / "model"
    steps = int(min(a.max_steps, max(600, a.epochs * n_tr / a.batch)))
    sh(f"cd /workspace/t2a && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_clap_v2.py --data {d} --out {out} --batch {a.batch} --max-steps {steps} --lr 1e-5 --text-lr 1e-5 --freeze-text-frac 0 --warmup 50 --eval-every 100 --eval-chunk 32 --save-every 100000 --push-every 100000 --total-rows {n_tr} > {a.out}/round{rnd:02d}/train.log 2>&1")
    evs = [json.loads(l) for l in open(out / "eval.jsonl")]
    best = max(evs, key=lambda e: e["macro_recall"]); final = evs[-1]
    return out, best, final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/workspace/rounds")); ap.add_argument("--pilot", type=Path, default=Path("/workspace/pilot")); ap.add_argument("--mel", type=Path, default=Path("/workspace/mel_all"))
    ap.add_argument("--start-per-class", type=int, default=240); ap.add_argument("--step-per-class", type=int, default=120); ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--model", default="gemini-3.1-pro-preview"); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--mel-workers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=24); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--eval-pct", type=int, default=20); ap.add_argument("--min-class", type=int, default=40); ap.add_argument("--head-frac", type=float, default=0.35)
    ap.add_argument("--target", type=float, default=0.85); ap.add_argument("--push-to", default="aoxo/clap-htsat-unfused-asmr-v2"); ap.add_argument("--first-round", type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    gm = GpuMel(); hist = []
    for rnd in range(a.first_round, a.rounds):
        per_class = a.start_per_class + a.step_per_class * rnd
        if rnd > a.first_round or per_class > 0: label_more(a, per_class)
        ledger = load_ledger(a); ensure_mel(a, ledger, gm)
        d, n_tr, n_ev = build_round_data(a, rnd)
        out, best, final = train_eval(a, rnd, d, n_tr)
        rec = {"round": rnd, "per_class": per_class, "labeled": len(ledger), "train": n_tr, "eval": n_ev, "final_top1": final["top1"], "final_macro": final["macro_recall"], "best_macro": best["macro_recall"], "best_step": best["step"], "best_top1": best["top1"], "recall": best["recall"]}
        hist.append(rec); (a.out / "rounds.jsonl").open("a").write(json.dumps(rec) + "\n")
        log(f"ROUND {rnd} labeled={len(ledger)} train={n_tr} eval={n_ev} top1={final['top1']:.3f} macro={final['macro_recall']:.3f} best_macro={best['macro_recall']:.3f}@{best['step']} recall={best['recall']}")
        try:
            from huggingface_hub import HfApi
            HfApi().upload_folder(folder_path=str(out), repo_id=a.push_to, repo_type="model", commit_message=f"vocal CLAP v2 round {rnd}: {len(ledger)} Pro-labeled clips, held-out top1={final['top1']:.3f} macro={final['macro_recall']:.3f}")
            log("pushed")
        except Exception as e: log(f"push failed: {str(e)[:120]}")
        if final["top1"] >= a.target and final["macro_recall"] >= a.target: log("TARGET_REACHED"); break
        if len(hist) >= 3 and max(h["best_macro"] for h in hist[-2:]) < hist[-3]["best_macro"] + 0.01: log("PLATEAU"); break
    log("ROUNDS_DONE")
if __name__ == "__main__": main()
