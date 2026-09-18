#!/usr/bin/env python3
"""Speech-gate the YouTube chapter windows: Silero VAD speech fraction + base-CLAP embedding per window, then linear probes at speech cutoffs.
Streams one FLAC at a time from aoxo/asmr-yt-chapters; resumable per video; low GPU footprint (fp16, small batches)."""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
SR = 48000; WIN = 4.0
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path.home() / "t2a/yt_vad"); ap.add_argument("--index", default="v2/yt_chapter_windows_index.jsonl")
    ap.add_argument("--repo", default="aoxo/asmr-yt-chapters"); ap.add_argument("--probe-only", action="store_true"); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    rows = [json.loads(l) for l in open(hf_hub_download("aoxo/clap-ft-data", a.index, repo_type="dataset")) if l.strip()]
    by_vid = defaultdict(list)
    for r in rows: by_vid[r["uid"].split(":")[1]].append(r)
    log(f"{len(rows)} windows across {len(by_vid)} videos")
    if not a.probe_only:
        from transformers import ClapModel, ClapFeatureExtractor
        from silero_vad import load_silero_vad, get_speech_timestamps
        fe = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused"); clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").cuda().half().eval(); vad = load_silero_vad()
        done = {p.stem for p in a.out.glob("*.npz")}
        for vid, wins in sorted(by_vid.items(), key=lambda kv: -len(kv[1])):
            if vid in done: continue
            local = None
            try:
                for i in range(4):
                    try: local = hf_hub_download(a.repo, f"audio/{vid}.flac", repo_type="dataset", local_dir=str(a.out / "tmp")); break
                    except Exception:
                        if i == 3: raise
                        time.sleep(15 * 2 ** i)
                out = subprocess.run(["ffmpeg", "-v", "error", "-threads", "4", "-i", local, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
                wav = np.frombuffer(out, dtype=np.float32); wav16 = torch.from_numpy(wav[::3].copy())  # 48k -> 16k by decimation (VAD only)
                uids, sp, embs = [], [], []
                for j in range(0, len(wins), a.batch):
                    ch = wins[j:j + a.batch]; segs = []
                    for r in ch:
                        s = float(r["uid"].split(":")[2]); seg = wav[int(s * SR): int((s + WIN) * SR)]
                        if len(seg) < SR: seg = np.pad(seg, (0, SR - len(seg)))
                        seg16 = wav16[int(s * 16000): int((s + WIN) * 16000)]
                        ts = get_speech_timestamps(seg16, vad, sampling_rate=16000, threshold=0.5, min_speech_duration_ms=100)
                        sp.append(sum(t["end"] - t["start"] for t in ts) / max(1, len(seg16))); segs.append(seg); uids.append(r["uid"])
                    feats = np.stack([fe(x, sampling_rate=SR, return_tensors="np")["input_features"][0].reshape(1001, 64) for x in segs])
                    with torch.no_grad():
                        x = torch.from_numpy(feats).half().unsqueeze(1).cuda()
                        e = clap.audio_model(input_features=x, is_longer=torch.zeros(len(ch), 1, dtype=torch.bool, device="cuda")).pooler_output
                    embs.append(e.float().cpu().numpy().astype(np.float16))
                np.savez(a.out / f"{vid}.npz", uids=np.array(uids), speech=np.array(sp, dtype=np.float32), emb=np.concatenate(embs))
                log(f"{vid}: {len(uids)} windows, median speech frac {np.median(sp):.2f}, done {len(list(a.out.glob('*.npz')))}/{len(by_vid)}")
            except Exception as e: log(f"FAIL {vid}: {type(e).__name__}: {str(e)[:120]}")
            finally:
                if local and os.path.exists(local): os.remove(local)
        torch.cuda.empty_cache()
    # ---- probes ----
    meta = {r["uid"]: r for r in rows}; U, S, E = [], [], []
    for p in a.out.glob("*.npz"):
        z = np.load(p); U += list(z["uids"]); S.append(z["speech"]); E.append(z["emb"])
    S = np.concatenate(S); E = torch.from_numpy(np.concatenate(E).astype(np.float32)); lab = [meta[u]["label"] for u in U]; vid = [u.split(":")[1] for u in U]
    log(f"probing on {len(U)} windows; speech-frac quantiles: {np.quantile(S, [0.25, 0.5, 0.75]).round(2).tolist()}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = random.Random(0)
    def probe(classes, keep, name, cap=800):
        idx = defaultdict(list)
        for i, (l, k) in enumerate(zip(lab, keep)):
            if k and l in classes: idx[l].append(i)
        tr, ev = [], []
        for c in classes:
            ii = idx[c]; vids = sorted({vid[i] for i in ii}); rng.shuffle(vids); evv = set(vids[: max(1, len(vids) // 4)])
            A = [i for i in ii if vid[i] not in evv]; B = [i for i in ii if vid[i] in evv]; rng.shuffle(A); rng.shuffle(B)
            tr += [(i, classes.index(c)) for i in A[:cap]]; ev += [(i, classes.index(c)) for i in B[: cap // 3]]
        if len(ev) < 20 or len({y for _, y in ev}) < len(classes): return log(f"  {name:48s} insufficient data ({len(tr)} tr / {len(ev)} ev)")
        Xtr = E[[i for i, _ in tr]].to(dev); Xev = E[[i for i, _ in ev]].to(dev); ytr = torch.tensor([y for _, y in tr], device=dev); yev = torch.tensor([y for _, y in ev], device=dev)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xev = (Xev - mu) / sd
        W = torch.zeros(Xtr.shape[1], len(classes), device=dev, requires_grad=True); b = torch.zeros(len(classes), device=dev, requires_grad=True); opt = torch.optim.Adam([W, b], lr=1e-2)
        for _ in range(600):
            loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr) + 1e-3 * (W ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): acc = ((Xev @ W + b).argmax(1) == yev).float().mean().item()
        maj = Counter(yev.tolist()).most_common(1)[0][1] / len(yev); log(f"  {name:48s} tr={len(tr):5d} ev={len(ev):4d} acc={acc:.3f} majority={maj:.3f} chance={1/len(classes):.3f}")
    PH = ["tapping", "scratching", "brushing", "crinkling", "liquid", "microphone touching", "sticky", "fabric rustling"]
    for cut in (1.01, 0.5, 0.3, 0.1, 0.0):
        keep = S <= cut if cut < 1 else np.ones(len(S), bool)
        if cut == 0.0: keep = S == 0.0
        log(f"== speech fraction <= {cut}: {int(keep.sum())} windows; per class {Counter(l for l, k in zip(lab, keep) if k and l in PH).most_common(8)}")
        probe(["tapping", "__bg__"], keep, "tapping vs whisper/talk-chapter windows"); probe(["tapping", "scratching"], keep, "tapping vs scratching"); probe(["tapping", "brushing"], keep, "tapping vs brushing")
        probe(["brushing", "scratching"], keep, "brushing vs scratching"); probe(["tapping", "crinkling"], keep, "tapping vs crinkling"); probe(PH, keep, "8-way physical")
    log("PROBE_DONE")
if __name__ == "__main__": main()
