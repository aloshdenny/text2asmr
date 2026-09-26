#!/usr/bin/env python3
"""YouTube chaptered ASMR -> labeled <=4s windows -> fp16 mel shards. Chapter title -> class via keyword map; loudness-gated; video-level split."""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time, zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_clap_v2 import GpuMel, repeatpad, ShardWriter, SR, FRAMES, MELS
BG = "__bg__"
CLASS_PAT = {  # order matters: first match wins only if exactly one class matches
    "tapping": r"\btap", "scratching": r"scratch", "crinkling": r"crinkl", "brushing": r"brush", "mouth sounds": r"mouth", "licking": r"lick",
    "kissing": r"kiss", "breathing": r"breath", "page turning": r"page turn|book|pages", "liquid": r"water|liquid|pour|spray|bubbl|fizz",
    "fabric rustling": r"fabric|cloth|leather|textile", "sticky": r"sticky|slime|gel", "cutting": r"\bcut|scissor", "microphone touching": r"\bmic\b|microphone",
    "hand movements": r"hand (movement|sound|motion)", "clinking": r"clink", "rustling": r"rustl", "paper rustling": r"paper", "blowing": r"\bblow", "typing": r"typing|keyboard", "writing": r"writing|pen\b|pencil",
}
BG_PAT = r"whisper|talk|rambl|chat|trigger word|story|roleplay|tk\b|sk\b|stipple|intro|outro|preview|gain|hello|goodbye|thank|sponsor|ad\b"
TEXTS = {c: [f"ASMR {c}, close-mic binaural recording, no speech", f"the sound of {c}", f"{c} sounds close to a microphone", f"soft {c} ASMR trigger"] for c in CLASS_PAT}
TEXTS["mouth sounds"] = ["ASMR mouth sounds, close-mic binaural, no speech", "soft mouth sounds", "lip smacking close to a microphone", "The sound of mouth sounds"]
TEXTS["kissing"] = ["ASMR kissing, close-mic binaural, no speech", "soft kisses close to a microphone", "lip kissing sounds", "The sound of kissing"]
TEXTS["breathing"] = ["ASMR breathing, close-mic binaural, no speech", "slow deep breathing", "soft breath close to a microphone", "heavy breathing", "The sound of breathing"]
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def classify(title: str):
    t = (title or "").lower(); hits = [c for c, p in CLASS_PAT.items() if re.search(p, t)]
    if len(hits) == 1 and not re.search(BG_PAT, t): return hits[0]
    if not hits and re.search(BG_PAT, t): return BG
    return None
def decode_span(path, start, dur):
    out = subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True, timeout=600).stdout
    return np.frombuffer(out, dtype=np.float32)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--repo", default="aoxo/asmr-yt-chapters")
    ap.add_argument("--win", type=float, default=4.0); ap.add_argument("--hop", type=float, default=3.0); ap.add_argument("--rms-db", type=float, default=-45.0)
    ap.add_argument("--per-chapter", type=int, default=40); ap.add_argument("--per-class", type=int, default=8000); ap.add_argument("--bg-per-video", type=int, default=40)
    ap.add_argument("--eval-pct", type=int, default=20); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--tmp", type=Path, default=Path("/workspace/tmp_yt"))
    # A pod is ephemeral: fp16 mel shards for 240 chapter-hours are ~37 GB and would die with it. In
    # manifest mode the pod does the expensive part (decode + loudness gate + chapter mapping) and emits a
    # few MB of window geometry, which any later run can cut mels from.
    ap.add_argument("--manifest-only", action="store_true", help="emit window rows, skip mel shards (no GPU needed)")
    ap.add_argument("--push-to", default="", help="HF dataset repo to upload the manifest to")
    ap.add_argument("--push-path", default="yt_windows/windows.jsonl")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); a.tmp.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); files = api.list_repo_files(a.repo, repo_type="dataset")
    metas = [f for f in files if f.startswith("meta/")]; audios = {f.split("/")[-1][:-5] for f in files if f.startswith("audio/")}
    vids = []
    for f in metas:
        vid = f.split("/")[-1].replace(".info.json", "")
        if vid not in audios: continue
        info = json.load(open(hf_hub_download(a.repo, f, repo_type="dataset")))
        chs = [{"start": c.get("start_time"), "end": c.get("end_time"), "cls": classify(c.get("title")), "title": c.get("title")} for c in (info.get("chapters") or [])]
        vids.append({"id": vid, "channel": info.get("channel") or info.get("uploader"), "chapters": [c for c in chs if c["cls"] and c["end"] and c["start"] is not None]})
    log(f"videos with audio+meta: {len(vids)}; labeled chapters: {sum(len(v['chapters']) for v in vids)}; class chapters: {Counter(c['cls'] for v in vids for c in v['chapters']).most_common()}")
    gm = None if a.manifest_only else GpuMel()
    w = None if a.manifest_only else ShardWriter(a.out, 4000)
    sink = None
    if a.manifest_only:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from text2asmr.io_guard import JsonlSink
        sink = JsonlSink(a.out / "windows.jsonl", key="uid")   # proves writability, resumes, flushes per row
        log(f"manifest mode: {len(sink.seen)} windows already recorded")
    per_class = Counter(); rng = np.random.default_rng(0)
    def one(v):
        local = None
        try:
            for i in range(4):
                try: local = hf_hub_download(a.repo, f"audio/{v['id']}.flac", repo_type="dataset", local_dir=str(a.tmp / v["id"])); break
                except Exception:
                    if i == 3: raise
                    time.sleep(10 * 2 ** i)
            split = "eval" if zlib.crc32(v["id"].encode()) % 100 < a.eval_pct else "train"; out = []
            for c in v["chapters"]:
                dur = c["end"] - c["start"]
                if dur < a.win: continue
                wav = decode_span(local, c["start"], min(dur, 900.0)); n = int((len(wav) / SR - a.win) / a.hop) + 1
                idx = list(range(max(0, n)));
                if len(idx) > (a.per_chapter if c["cls"] != BG else a.bg_per_video): idx = sorted(rng.choice(idx, a.per_chapter if c["cls"] != BG else a.bg_per_video, replace=False).tolist())
                for k in idx:
                    seg = wav[int(k * a.hop * SR): int(k * a.hop * SR + a.win * SR)]
                    if len(seg) < SR: continue
                    db = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)
                    if db < a.rms_db: continue
                    meta = {"uid": f"yt:{v['id']}:{c['start'] + k * a.hop:.1f}", "label": c["cls"],
                            "text": TEXTS.get(c["cls"], []), "split": split, "source": f"yt:{v['id']}",
                            "chapter": c["title"], "rms_db": round(float(db), 1),
                            "start": round(c["start"] + k * a.hop, 3), "dur": a.win, "repo": a.repo}
                    out.append((None if a.manifest_only else repeatpad(seg), meta))
            return v["id"], out
        except Exception as e: log(f"video fail {v['id']}: {type(e).__name__}: {str(e)[:120]}"); return v["id"], []
        finally:
            if local:
                try: os.remove(local)
                except Exception: pass
    done_v = 0; n_rows = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for vid, out in ex.map(one, vids):
            keep = [(x, m) for x, m in out if per_class[m["label"]] < a.per_class or m["split"] == "eval"]
            for _, m in keep: per_class[m["label"]] += 1
            if a.manifest_only:
                for _, m in keep: sink.write(m)
            else:
                for i in range(0, len(keep), 128):
                    ch = keep[i:i+128]; w.write(gm(np.stack([x for x, _ in ch])), [m for _, m in ch])
            done_v += 1; n_rows += len(keep); log(f"{done_v}/{len(vids)} {vid} +{len(keep)} rows (total {n_rows})")
    if sink: sink.close()
    if a.push_to:
        for attempt in range(5):
            try:
                api.upload_file(path_or_fileobj=str(a.out / "windows.jsonl"), path_in_repo=a.push_path,
                                repo_id=a.push_to, repo_type="dataset",
                                commit_message=f"YouTube chapter windows: {n_rows} rows, {len(per_class)} classes")
                log(f"uploaded {a.push_path} -> {a.push_to}"); break
            except Exception as e:
                log(f"upload retry {attempt}: {type(e).__name__} {str(e)[:110]}"); time.sleep(30 * (attempt + 1))
    log(f"PREP_DONE videos={done_v} rows={n_rows} per_class={dict(per_class)}")
if __name__ == "__main__": main()
