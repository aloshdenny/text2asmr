#!/usr/bin/env python3
"""Fine-tune CLAP with live manifest refresh (new Gemini labels mid-run).

Mac side periodically rebuilds + uploads label_tool/clap_finetune_manifest.jsonl
to HF dataset aoxo/clap-ft-data. This trainer re-pulls every --sync-every steps
and appends unseen uids into the active pool without stopping.

  python3 scripts/finetune_clap.py \\
    --manifest-repo aoxo/clap-ft-data \\
    --manifest-file clap_finetune_manifest.jsonl \\
    --out /workspace/clap-ft --batch 16 --lr 1e-5 \\
    --sync-every 40 --save-every 100 --max-steps 4000 \\
    --push-to aoxo/clap-htsat-unfused-asmr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CLAP_SR = 48_000
MAX_SAMPLES = CLAP_SR * 10
BASE_MODEL = os.environ.get("CLAP_BASE_MODEL", "laion/clap-htsat-unfused")
DEFAULT_MANIFEST_REPO = "aoxo/clap-ft-data"
DEFAULT_MANIFEST_FILE = "clap_finetune_manifest.jsonl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hf_token() -> str:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not set")
    return token


def decode_path(path: Path, sr: int = CLAP_SR, mono: bool = True,
                start: float | None = None, dur: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-threads", "1"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += [
        "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr), "-ac", "1" if mono else "2", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(out, dtype=np.float32).copy()
    if mono:
        return audio
    return audio.reshape(-1, 2).T if audio.size else audio.reshape(2, 0)


def wav_cache_path(cache: Path, uid: str) -> Path:
    digest = hashlib.sha1(uid.encode()).hexdigest()[:16]
    return cache / "wav" / digest[:2] / f"{digest}.npy"


def feat_cache_path(cache: Path, uid: str) -> Path:
    digest = hashlib.sha1(uid.encode()).hexdigest()[:16]
    return cache / "feat" / digest[:2] / f"{digest}.pt"


def resolve_audio(row: dict, cache: Path) -> np.ndarray | None:
    """Decode once, then reload from local .npy (GPU was starved by ffmpeg)."""
    from huggingface_hub import hf_hub_download

    uid = str(row.get("uid") or "")
    npy = wav_cache_path(cache, uid) if uid else None
    if npy is not None and npy.exists():
        try:
            return np.load(npy, mmap_mode=None)
        except Exception:
            pass

    audio = row["audio"]
    kind = audio["kind"]
    try:
        if kind == "segments_flac":
            p = Path(hf_hub_download(
                audio["repo"], audio["path"], repo_type="dataset",
                token=hf_token(), cache_dir=str(cache / "hf"),
            ))
            wav = decode_path(p)
        elif kind == "audios2_cut":
            p = Path(hf_hub_download(
                audio["repo"], audio["source"], repo_type="dataset",
                token=hf_token(), cache_dir=str(cache / "hf"),
            ))
            wav = decode_path(p, start=float(audio["start"]), dur=float(audio["duration"]))
        else:
            raise ValueError(f"unknown audio kind {kind}")
    except Exception as exc:
        log(f"audio skip {row.get('uid')}: {type(exc).__name__}: {exc}")
        return None

    if wav.size > MAX_SAMPLES:
        wav = wav[:MAX_SAMPLES]
    if npy is not None:
        npy.parent.mkdir(parents=True, exist_ok=True)
        tmp = npy.with_suffix(".tmp.npy")
        np.save(tmp, wav)
        tmp.replace(npy)
    return wav


def resolve_audio_features(row: dict, cache: Path, processor) -> "object":
    """Cache CLAP input_features so the train loop never waits on mel extract."""
    import torch

    uid = str(row.get("uid") or "")
    path = feat_cache_path(cache, uid) if uid else None
    if path is not None and path.exists():
        try:
            t = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            try:
                t = torch.load(path, map_location="cpu")
            except Exception:
                t = None
        if t is not None:
            if t.dim() == 2:
                t = t.unsqueeze(0).contiguous()
            return t

    wav = resolve_audio(row, cache)
    if wav is None or wav.size < CLAP_SR // 4:
        wav = np.zeros(CLAP_SR, dtype=np.float32)
    if wav.size > MAX_SAMPLES:
        wav = wav[:MAX_SAMPLES]
    feats = processor(audios=wav, sampling_rate=CLAP_SR, return_tensors="pt")
    tensor = feats["input_features"][0].contiguous()
    # Keep channel dim so stacked batch is (B, 1, T, F) — CLAP indexes dim 3.
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).contiguous()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.pt")
        torch.save(tensor, tmp)
        tmp.replace(path)
    return tensor


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def pull_manifest(repo: str, filename: str, dest: Path, force: bool = True) -> Path:
    from huggingface_hub import hf_hub_download

    # Bust HF cache so mid-run uploads are visible.
    p = hf_hub_download(
        repo, filename, repo_type="dataset", token=hf_token(),
        force_download=force,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Local manifest path (optional if --manifest-repo set)")
    ap.add_argument("--manifest-repo", default=DEFAULT_MANIFEST_REPO)
    ap.add_argument("--manifest-file", default=DEFAULT_MANIFEST_FILE)
    ap.add_argument("--out", type=Path, default=Path("/workspace/clap-ft"))
    ap.add_argument("--cache", type=Path, default=Path("/workspace/clap_ft_cache"))
    ap.add_argument("--init", type=Path, default=None,
                    help="Resume weights dir (defaults to --out if config.json present)")
    ap.add_argument("--epochs", type=int, default=0,
                    help="Legacy epoch budget; 0 = use --max-steps only")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--sync-every", type=int, default=80,
                    help="Re-pull Hub manifest and ingest new uids every N steps")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel ffmpeg/HF decode workers for batch build")
    ap.add_argument("--prefetch", type=int, default=4,
                    help="Batches to prepare ahead of the GPU step")
    ap.add_argument("--feeders", type=int, default=3,
                    help="CPU feeder threads building batches in parallel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--push-to", default="")
    args = ap.parse_args()

    import torch
    from transformers import ClapModel, ClapProcessor

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    args.cache.mkdir(parents=True, exist_ok=True)
    (args.cache / "wav").mkdir(parents=True, exist_ok=True)
    (args.cache / "feat").mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    local_manifest = args.manifest or (HERE / "label_tool" / args.manifest_file)

    known: dict[str, dict] = {}

    def sync_pool(force: bool = False) -> int:
        """Pull latest manifest; return count of newly added uids."""
        try:
            pull_manifest(args.manifest_repo, args.manifest_file, local_manifest, force=force)
        except Exception as exc:
            log(f"manifest pull failed: {type(exc).__name__}: {exc}")
            if not local_manifest.exists():
                return 0
        added = 0
        for row in load_manifest(local_manifest):
            uid = row.get("uid")
            if not uid or uid in known:
                continue
            known[uid] = row
            added += 1
        if added:
            log(f"ingested +{added} clips (pool={len(known)})")
        return added

    sync_pool(force=True)
    if len(known) < args.batch:
        raise SystemExit(f"need >= {args.batch} clips, got {len(known)}")

    init_path = args.init
    if init_path is None and (args.out / "config.json").exists():
        init_path = args.out
    start_from = str(init_path) if init_path else BASE_MODEL
    resume_step = 0
    meta_path = Path(start_from) / "finetune_meta.json" if init_path else None
    if meta_path and meta_path.exists():
        try:
            resume_step = int(json.loads(meta_path.read_text()).get("step") or 0)
        except Exception:
            resume_step = 0
    log(f"init_weights={start_from} resume_step={resume_step} pool={len(known)} "
        f"batch={args.batch} workers={args.workers} feeders={args.feeders} "
        f"sync_every={args.sync_every}")

    processor = ClapProcessor.from_pretrained(BASE_MODEL)
    model = ClapModel.from_pretrained(start_from).to(args.device)
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )

    def contrastive_loss(audio_emb, text_emb):
        audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
        if hasattr(model, "logit_scale"):
            scale = model.logit_scale.exp()
        elif hasattr(model, "logit_scale_a") and hasattr(model, "logit_scale_t"):
            scale = (model.logit_scale_a.exp() + model.logit_scale_t.exp()) / 2
        else:
            scale = torch.tensor(1.0 / 0.07, device=audio_emb.device)
        logits = scale * audio_emb @ text_emb.T
        labels = torch.arange(logits.size(0), device=logits.device)
        return (
            torch.nn.functional.cross_entropy(logits, labels)
            + torch.nn.functional.cross_entropy(logits.T, labels)
        ) / 2

    pool = ThreadPoolExecutor(max_workers=max(2, args.workers))
    from queue import Queue, Empty, Full
    import threading

    hot: set[str] = set()
    hot_lock = threading.Lock()
    for uid in known:
        if feat_cache_path(args.cache, uid).exists():
            hot.add(uid)
    log(f"hot feat cache ready={len(hot)}/{len(known)}")

    def _one_feat(uid: str):
        row = known[uid]
        feat = resolve_audio_features(row, args.cache, processor)
        with hot_lock:
            hot.add(uid)
        cand = row.get("text") or [row.get("caption") or row["trigger"]]
        return feat, random.choice(cand)

    def sample_batch_tensors():
        with hot_lock:
            hot_list = list(hot)
        if len(hot_list) >= args.batch:
            pick = random.sample(hot_list, args.batch)
        else:
            uids = list(known.keys())
            if len(uids) < args.batch:
                return None
            pick = random.sample(uids, args.batch)
        parts = list(pool.map(_one_feat, pick))
        feats = [f for f, _ in parts]
        texts = [t for _, t in parts]
        norm = []
        for f in feats:
            if f.dim() == 2:
                f = f.unsqueeze(0)
            norm.append(f)
        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
        audio_inputs = {
            "input_features": torch.stack(norm, dim=0),
            "is_longer": torch.zeros(len(norm), 1, dtype=torch.bool),
        }
        if args.device.startswith("cuda"):
            text_inputs = {k: v.pin_memory() if hasattr(v, "pin_memory") else v
                           for k, v in text_inputs.items()}
            audio_inputs = {k: v.pin_memory() if hasattr(v, "pin_memory") else v
                            for k, v in audio_inputs.items()}
        return text_inputs, audio_inputs

    batch_q: Queue = Queue(maxsize=max(4, args.prefetch))
    stop_flag = threading.Event()

    def feeder() -> None:
        while not stop_flag.is_set():
            try:
                item = sample_batch_tensors()
                while not stop_flag.is_set():
                    try:
                        batch_q.put(item, timeout=1.0)
                        break
                    except Full:
                        continue
            except Exception as exc:
                log(f"feeder err: {type(exc).__name__}: {exc}")
                time.sleep(1.0)

    def warmer() -> None:
        """Background: fill wav+feat caches so steady-state GPU stays saturated."""
        uids = list(known.keys())
        random.shuffle(uids)
        done = 0
        for uid in uids:
            if stop_flag.is_set():
                return
            try:
                resolve_audio_features(known[uid], args.cache, processor)
            except Exception:
                pass
            done += 1
            if done % 500 == 0:
                log(f"cache warm {done}/{len(uids)}")
        log(f"cache warm complete {done}/{len(uids)}")

    for i in range(max(1, args.feeders)):
        threading.Thread(target=feeder, name=f"clap-feeder-{i}", daemon=True).start()
    threading.Thread(target=warmer, name="clap-warmer", daemon=True).start()

    def save(step: int, loss: float, tag: str = "ckpt") -> None:
        model.save_pretrained(args.out)
        processor.save_pretrained(args.out)
        meta = {
            "base": BASE_MODEL,
            "init": start_from,
            "step": step,
            "pool": len(known),
            "loss": loss,
            "lr": args.lr,
            "batch": args.batch,
            "tag": tag,
        }
        (args.out / "finetune_meta.json").write_text(json.dumps(meta, indent=2))
        log(f"saved {tag} step={step} pool={len(known)} loss={loss:.4f} -> {args.out}")

    t0 = time.time()
    running = 0.0
    n = 0
    step = resume_step
    target = args.max_steps
    if args.epochs > 0:
        target = max(target, args.epochs * max(1, len(known) // args.batch))

    log(f"dynamic train start step={step} max_steps={target}")
    gpu_busy = []
    while step < target:
        if args.sync_every and step > 0 and step % args.sync_every == 0:
            sync_pool(force=(step % (args.sync_every * 5) == 0))
            target = max(target, step + max(200, len(known) // args.batch))

        try:
            batch = batch_q.get(timeout=180)
        except Empty:
            log("batch queue empty; syncing")
            sync_pool(force=True)
            continue
        if batch is None:
            log("pool too small; syncing and waiting")
            sync_pool(force=True)
            time.sleep(5)
            continue
        text_inputs, audio_inputs = batch

        t_gpu0 = time.time()
        text_inputs = {k: v.to(args.device, non_blocking=True) for k, v in text_inputs.items()}
        audio_inputs = {k: v.to(args.device, non_blocking=True) for k, v in audio_inputs.items()}

        text_emb = model.get_text_features(**text_inputs)
        audio_emb = model.get_audio_features(**audio_inputs)
        if hasattr(text_emb, "pooler_output"):
            text_emb = text_emb.pooler_output
        if hasattr(audio_emb, "pooler_output"):
            audio_emb = audio_emb.pooler_output
        loss = contrastive_loss(audio_emb, text_emb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        gpu_busy.append(time.time() - t_gpu0)

        step += 1
        n += 1
        running += float(loss.detach())
        if step % 20 == 0:
            gb = sum(gpu_busy[-20:]) / max(1, len(gpu_busy[-20:]))
            log(f"train step={step} loss={running/n:.4f} pool={len(known)} "
                f"batch={args.batch} gpu_s/step={gb:.2f} q={batch_q.qsize()} "
                f"elapsed_min={(time.time()-t0)/60:.1f}")
        if args.save_every and step % args.save_every == 0:
            save(step, running / n)

    stop_flag.set()
    pool.shutdown(wait=False, cancel_futures=True)
    save(step, running / max(n, 1), tag="final")
    if args.push_to:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token())
        api.create_repo(args.push_to, repo_type="model", exist_ok=True, private=True)
        api.upload_folder(folder_path=str(args.out), repo_id=args.push_to, repo_type="model")
        log(f"pushed -> {args.push_to}")

    (args.out / "CHECKPOINT_DONE").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    log("CHECKPOINT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
