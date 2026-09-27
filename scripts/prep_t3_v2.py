#!/usr/bin/env python3
"""Generator v2 data prep: speech windows -> S3 speech tokens + prompt tokens + speaker embedding.

Streams recording by recording so the source audio never accumulates (2,000 h of windows come from ~18k
recordings, ~270 GB of m4a -- more than the disk holds). Per recording: download -> decode once to 16 kHz ->
tokenise every window and its 4 s continuation prompt on the GPU -> keep only the tokens -> delete the audio.

Output: .pt shards of rows {uid, text, speech (int16), prompt (int16 | None), spk (float16[256]), dur, creator}.
Everything a T3 training step needs, so training never touches audio. Resumable (done list), and each shard is
pushed to the Hub as it closes.

  python prep_t3_v2.py --manifest D:\\t2a\\speech_windows_v2.jsonl --out D:\\t2a\\t3v2 --push aoxo/t2a-speech-v2
"""
from __future__ import annotations
import argparse, json, os, queue, subprocess, sys, threading, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text2asmr.models.chatterbox_ft import load_backbone

SR = 16_000


def log(m): print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def decode16(path: str) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True, timeout=900).stdout
    return np.frombuffer(out, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--push", default="", help="HF dataset repo for finished shards")
    ap.add_argument("--shard-rows", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=24, help="windows per S3-tokenizer call")
    ap.add_argument("--prefetch", type=int, default=6, help="recordings downloaded ahead of the GPU")
    ap.add_argument("--dl-workers", type=int, default=4)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, hf_hub_download
    a.out.mkdir(parents=True, exist_ok=True)
    done_path = a.out / "done_recordings.txt"
    done = set(done_path.read_text(encoding="utf-8").split("\n")) if done_path.exists() else set()

    by_rec: dict[tuple, list] = defaultdict(list)
    for l in a.manifest.open(encoding="utf-8"):
        r = json.loads(l)
        by_rec[(r["repo"], r["source"])].append(r)
    todo = [k for k in by_rec if f"{k[0]}|{k[1]}" not in done]
    log(f"{sum(len(v) for v in by_rec.values())} windows in {len(by_rec)} recordings; {len(todo)} recordings to do")

    model = load_backbone("cuda")
    tok, ve = model.s3gen.tokenizer, model.ve
    api = HfApi() if a.push else None
    if api: api.create_repo(a.push, repo_type="dataset", exist_ok=True)

    # ---- network: download ahead of the GPU, bounded so the disk never fills ----
    q: queue.Queue = queue.Queue(maxsize=a.prefetch)
    it = iter(todo); lock = threading.Lock()

    def downloader():
        while True:
            with lock:
                k = next(it, None)
            if k is None: q.put(None); return
            repo, src = k
            for i in range(4):
                try:
                    p = hf_hub_download(repo, src, repo_type="dataset", local_dir=str(a.out / "dl"))
                    q.put((k, p)); break
                except Exception as e:
                    if i == 3: log(f"  download failed {src}: {type(e).__name__}"); q.put((k, None))
                    else: time.sleep(10 * 2 ** i)
    for _ in range(a.dl_workers): threading.Thread(target=downloader, daemon=True).start()

    shard_rows, shard_i = [], len(list(a.out.glob("shard_*.pt")))
    pending: list[str] = []                             # recordings whose rows are in memory, not yet on disk
    n_done, n_rows, t0, finished = 0, 0, time.time(), 0

    def flush(final=False):
        nonlocal shard_rows, shard_i
        if not shard_rows or (len(shard_rows) < a.shard_rows and not final): return
        p = a.out / f"shard_{shard_i:05d}.pt"
        torch.save(shard_rows, p)
        if api:
            for i in range(5):
                try:
                    api.upload_file(path_or_fileobj=str(p), path_in_repo=f"t3v2/{p.name}", repo_id=a.push, repo_type="dataset",
                                    commit_message=f"t3 v2 prep: {p.name} ({len(shard_rows)} windows)")
                    break
                except Exception as e:
                    log(f"  push retry {i}: {type(e).__name__}"); time.sleep(30 * (i + 1))
        log(f"  wrote {p.name}: {len(shard_rows)} windows")
        # only now are these recordings safe: marking them done earlier would lose them on a crash
        with done_path.open("a", encoding="utf-8") as fh: fh.write("".join(k + "\n" for k in pending))
        pending.clear()
        shard_rows, shard_i = [], shard_i + 1

    while finished < a.dl_workers:
        item = q.get()
        if item is None: finished += 1; continue
        (repo, src), path = item
        rows = by_rec[(repo, src)]
        if path:
            try:
                wav = decode16(path)
                segs, meta = [], []
                for r in rows:
                    w = wav[int(r["start"] * SR): int(r["end"] * SR)]
                    if len(w) < SR: continue
                    pr = wav[int(r["prompt_start"] * SR): int(r["prompt_end"] * SR)] if r.get("prompt_start") is not None else None
                    segs.append(w); meta.append((r, pr))
                for b in range(0, len(segs), a.batch):
                    ws = segs[b:b + a.batch]; ms = meta[b:b + a.batch]
                    with torch.no_grad():
                        st, sl = tok.forward(ws)
                        prs = [m[1] for m in ms if m[1] is not None and len(m[1]) >= SR]
                        pt, pl = tok.forward(prs) if prs else (None, None)
                        spk = np.asarray(ve.embeds_from_wavs(ws, sample_rate=SR))
                    pi = 0
                    for k, (r, pr) in enumerate(ms):
                        prompt = None
                        if pr is not None and len(pr) >= SR:
                            prompt = pt[pi, :pl[pi]].cpu().to(torch.int16); pi += 1
                        shard_rows.append({"uid": r["uid"], "text": r["text"], "dur": r["dur"], "creator": r.get("creator"),
                                           "repo": repo, "speech": st[k, :sl[k]].cpu().to(torch.int16),
                                           "prompt": prompt, "spk": torch.from_numpy(spk[k]).to(torch.float16)})
                        n_rows += 1
                    flush()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); log(f"  OOM on {src}; skipped")
            except Exception as e:
                log(f"  {src}: {type(e).__name__} {str(e)[:120]}")
            finally:
                try: os.remove(path)
                except OSError: pass
        pending.append(f"{repo}|{src}")
        n_done += 1
        if n_done % 25 == 0:
            el = time.time() - t0
            log(f"{n_done}/{len(todo)} recordings, {n_rows} windows, {n_rows / max(el, 1) * 3600:.0f} windows/h, "
                f"ETA {(len(todo) - n_done) * el / n_done / 3600:.1f} h")
    flush(final=True)
    log(f"PREP_DONE recordings={n_done} windows={n_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
