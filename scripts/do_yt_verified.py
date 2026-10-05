#!/usr/bin/env python3
"""DO droplet: download -> verify -> non-speech map -> upload, one video at a time.

Replaces the plain downloader (do_yt_pipeline.py) as the droplet's only YouTube fetcher, so the politeness
budget is shared, not doubled. Order of work:
  1. candidates from discovery (yt_candidates.jsonl: id, cls, kind single|chapter, chapters)
  2. the legacy chaptered queue (yt_remaining.txt), labelled later from info.json chapters

Verify, before a byte is uploaded:
  * the file decodes and its duration matches what YouTube reported (+-5%)
  * the labelled span is not silence (median 1 s RMS above --min-db)
  * Silero VAD speech ratio over the labelled span is below --max-speech. Discovery trusts titles like
    "no talking"; this checks them. A failed video is recorded in rejects and never uploaded.

The VAD pass also writes the non-speech map (speech intervals per video) to vad/<id>.json. It is the cheap
stand-in for a first transcription pass: the gaps between speech are where triggers live, and Whisper on a
GPU pod later re-transcribes only what matters. Whisper does not run here: the droplet is 1 vCPU / 1 GB.

Throttling is respected, never evaded: at most T2A_YT_PER_HOUR fetches in any rolling hour, and every
block ("Sign in to confirm", 429, ...) doubles a sleep that a success resets.

  python3 do_yt_verified.py --candidates /root/t2a/yt_candidates.jsonl --legacy /root/t2a/yt_remaining.txt
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_filter import blocked_title

REPO = "aoxo/asmr-yt-chapters"
SR = 16_000
BLOCK_MARKERS = ("Sign in to confirm", "HTTP Error 429", "Too Many Requests", "blocked it in your country",
                 "This content isn't available", "The page needs to be reloaded")


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


class Vad:
    """Silero VAD v5 through onnxruntime directly: ~50 MB RAM instead of the ~700 MB torch would cost."""
    def __init__(self, model: Path):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(str(model), sess_options=so, providers=["CPUExecutionProvider"])

    def speech(self, path: Path, start: float, end: float, thr: float = 0.5) -> tuple[list, float, float]:
        """Stream [start, end) through the model. Returns speech intervals (absolute s), speech ratio and the
        median 1 s RMS in dB (the silence check rides on the same decode)."""
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(path),
                              "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], stdout=subprocess.PIPE)
        state = np.zeros((2, 1, 128), np.float32); ctx = np.zeros((1, 64), np.float32)
        sr = np.array(SR, dtype=np.int64)
        probs, rms, sec = [], [], []
        while True:
            buf = p.stdout.read(512 * 4)
            if len(buf) < 512 * 4: break
            x = np.frombuffer(buf, np.float32)[None, :]
            out, state = self.s.run(None, {"input": np.concatenate([ctx, x], 1), "state": state, "sr": sr})
            ctx = x[:, -64:]
            probs.append(float(out[0, 0]))
            sec.append(x[0])
            if len(sec) == 31:                          # ~1 s of 32 ms frames
                s = np.concatenate(sec); rms.append(20 * np.log10(np.sqrt(np.mean(s * s)) + 1e-9)); sec = []
        p.wait()
        on = np.array(probs) > thr
        segs, i, hop = [], 0, 512 / SR
        while i < len(on):                              # frames -> intervals; drop blips, bridge short gaps
            if on[i]:
                j = i
                while j < len(on) and on[j]: j += 1
                segs.append([start + i * hop, start + j * hop]); i = j
            else: i += 1
        merged = []
        for s, e in segs:
            if merged and s - merged[-1][1] < 0.3: merged[-1][1] = e
            else: merged.append([s, e])
        merged = [[round(s, 2), round(e, 2)] for s, e in merged if e - s >= 0.25]
        ratio = sum(e - s for s, e in merged) / max(end - start, 1e-6)
        return merged, ratio, float(np.median(rms)) if rms else -120.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=Path("/root/t2a/yt_candidates.jsonl"))
    ap.add_argument("--legacy", type=Path, default=Path("/root/t2a/yt_remaining.txt"))
    ap.add_argument("--work", type=Path, default=Path("/root/t2a/ytv"))
    ap.add_argument("--cookies", type=Path, default=Path("/root/t2a/cookies.txt"))
    ap.add_argument("--vad-model", type=Path, default=Path("/root/t2a/models/silero_vad.onnx"))
    ap.add_argument("--max-speech", type=float, default=0.15, help="max VAD speech ratio in a no-talking span")
    ap.add_argument("--min-db", type=float, default=-65.0, help="median 1 s RMS below this = silence (quiet triggers like page turning sit at -55..-63)")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    a = ap.parse_args()
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(); a.work.mkdir(parents=True, exist_ok=True)
    per_hour = int(os.environ.get("T2A_YT_PER_HOUR", "12"))
    vad = Vad(a.vad_model)
    done_path, rej_path = a.work / "done.txt", a.work / "rejects.jsonl"
    done = set(done_path.read_text().split()) if done_path.exists() else set()

    penalty, recent = 0, []
    while True:
        # re-read every pass: discovery keeps appending candidates while this runs
        have = {f.split("/")[1].rsplit(".", 1)[0] for f in api.list_repo_files(REPO, repo_type="dataset")
                if f.startswith("audio/")}
        queue = []
        if a.candidates.exists():
            for l in a.candidates.open():
                try: c = json.loads(l)
                except Exception: continue
                if blocked_title(c.get("title") or ""): continue      # content filter (content_filter.py)
                queue.append(c)
        if a.legacy.exists():
            for u in a.legacy.read_text().split():
                m = re.search(r"v=([\w-]{11})", u)
                if m: queue.append({"id": m.group(1), "url": u, "kind": "legacy"})
        seen, todo = set(), []
        for c in queue:
            if c["id"] in seen or c["id"] in done or c["id"] in have: continue
            seen.add(c["id"]); todo.append(c)
        if not todo:
            log("queue empty; checking again in 30 min"); time.sleep(1800); continue
        log(f"{len(todo)} to fetch ({sum(c['kind'] != 'legacy' for c in todo)} discovered, "
            f"{sum(c['kind'] == 'legacy' for c in todo)} legacy); {per_hour}/h")

        cand_mtime = a.candidates.stat().st_mtime if a.candidates.exists() else 0
        for c in todo:
            # discovery keeps merging new candidates; they outrank the legacy queue, so re-plan when they land
            if (a.candidates.stat().st_mtime if a.candidates.exists() else 0) != cand_mtime:
                log("candidate list changed; re-planning"); break
            if shutil.disk_usage("/").free / 1e9 < a.min_free_gb:
                log("low disk; pausing 30 min"); time.sleep(1800); break
            now = time.time(); recent[:] = [t for t in recent if now - t < 3600]
            if len(recent) >= per_hour:
                wait = 3600 - (now - recent[0]) + 5
                log(f"hourly ceiling reached; waiting {wait / 60:.0f} min"); time.sleep(max(wait, 60))
            recent.append(time.time())
            vid = c["id"]; flac, info = a.work / f"{vid}.flac", a.work / f"{vid}.info.json"
            r = subprocess.run(["/root/t2a/venv/bin/yt-dlp", "--cookies", str(a.cookies), "-f", "bestaudio/best", "-x",
                                "--audio-format", "flac", "--audio-quality", "0", "--write-info-json",
                                "--sleep-requests", "1.5", "--sleep-interval", "30", "--max-sleep-interval", "120",
                                "--no-part", "-o", f"{a.work}/%(id)s.%(ext)s", c.get("url") or f"https://www.youtube.com/watch?v={vid}"],
                               capture_output=True, text=True)
            err = (r.stderr or "")[-400:]
            if any(m in err for m in BLOCK_MARKERS):
                penalty = 1800 if not penalty else min(penalty * 2, 4 * 3600)
                log(f"{vid}: blocked ({err.strip().splitlines()[-1][:80]}); backing off {penalty / 60:.0f} min")
                for f in (flac, info): f.unlink(missing_ok=True)
                time.sleep(penalty); continue
            if r.returncode or not flac.exists() or not info.exists():
                log(f"{vid}: download failed: {err.strip().splitlines()[-1][:100] if err.strip() else r.returncode}")
                with rej_path.open("a") as fh: fh.write(json.dumps({"id": vid, "reason": "download", "err": err[-200:]}) + "\n")
                done.add(vid); done_path.write_text("\n".join(sorted(done)))
                for f in (flac, info): f.unlink(missing_ok=True)
                continue
            penalty = 0

            # ---- verify ----
            meta = json.loads(info.read_text())
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                        str(flac)], capture_output=True, text=True).stdout.strip() or 0)
            want = float(meta.get("duration") or c.get("duration_s") or 0)
            reason = None
            if dur <= 0 or (want and abs(dur - want) > 0.05 * want + 5): reason = f"duration {dur:.0f}s vs {want:.0f}s"
            spans = [(ch["start"], ch["end"]) for ch in c.get("chapters") or []] if c["kind"] == "chapter" else \
                    [(min(60.0, dur * 0.05), max(dur - 60.0, dur * 0.95))] if c["kind"] == "single" else [(0.0, dur)]
            vad_out, ratios, dbs = [], [], []
            if not reason:
                for s, e in spans:
                    e = min(e, dur)
                    if e - s < 30: continue
                    segs, ratio, db = vad.speech(flac, s, e)
                    vad_out += segs; ratios.append((ratio, e - s)); dbs.append(db)
                sp = sum(r_ * w for r_, w in ratios) / max(sum(w for _, w in ratios), 1e-6) if ratios else 0.0
                if c["kind"] in ("single", "chapter"):
                    if not ratios: reason = "no usable labelled span"
                    elif sp > a.max_speech: reason = f"speech {sp:.0%} in a no-talking span"
                    elif np.median(dbs) < a.min_db: reason = f"silent (median {np.median(dbs):.0f} dB)"
            verdict = {"id": vid, "cls": c.get("cls"), "kind": c["kind"], "chapters": c.get("chapters") or [],
                       "duration_s": round(dur, 1), "channel": meta.get("channel") or meta.get("uploader"),
                       "speech_ratio": round(sp, 3) if ratios else None, "median_db": round(float(np.median(dbs)), 1) if dbs else None,
                       "source": "discovery" if c["kind"] != "legacy" else "legacy", "verified_at": time.strftime("%F %T")}
            if reason:
                log(f"{vid}: REJECT {c.get('cls') or ''} {reason}")
                with rej_path.open("a") as fh: fh.write(json.dumps(dict(verdict, reason=reason)) + "\n")
                done.add(vid); done_path.write_text("\n".join(sorted(done)))
                for f in (flac, info): f.unlink(missing_ok=True)
                continue

            # ---- upload: audio, metadata, non-speech map and label sidecar in ONE commit ----
            vad_json = a.work / f"{vid}.vad.json"
            vad_json.write_text(json.dumps({"id": vid, "speech": vad_out, "spans": spans, "sr": SR, "model": "silero_vad_v5"}))
            lab_json = a.work / f"{vid}.label.json"
            lab_json.write_text(json.dumps(verdict))
            ops = [CommitOperationAdd(f"audio/{vid}.flac", str(flac)), CommitOperationAdd(f"meta/{vid}.info.json", str(info)),
                   CommitOperationAdd(f"vad/{vid}.json", str(vad_json))]
            if c["kind"] != "legacy": ops.append(CommitOperationAdd(f"labels/video/{vid}.json", str(lab_json)))
            for attempt in range(5):
                try:
                    api.create_commit(REPO, repo_type="dataset", operations=ops,
                                      commit_message=f"{vid}: {c.get('cls') or 'chaptered'} ({c['kind']}), verified")
                    break
                except Exception as e:
                    log(f"{vid}: upload attempt {attempt + 1} failed: {type(e).__name__} {str(e)[:80]}"); time.sleep(30 * (attempt + 1))
            else:
                log(f"{vid}: upload gave up; keeping files for the next pass"); continue
            log(f"{vid}: OK {c.get('cls') or 'legacy'} {dur / 60:.0f} min, speech {verdict['speech_ratio']}, "
                f"{verdict['median_db']} dB, {len(vad_out)} speech segs")
            done.add(vid); done_path.write_text("\n".join(sorted(done)))
            for f in (flac, info, vad_json, lab_json): f.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
