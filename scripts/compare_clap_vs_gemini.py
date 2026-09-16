#!/usr/bin/env python3
"""Compare original CLAP / base unfused / fine-tuned CLAP against Gemini labels.

Uses label_tool/gemini_retag.jsonl rows that already have both clap_label and
gemini_label, downloads triggers from aoxo/text2asmr-segments, and scores with
the same ontology probe bank as clap_label_audios2.py.

  source ~/.t2a_env
  PYTHONPATH=. python3 scripts/compare_clap_vs_gemini.py --limit 600 --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SEGMENTS_REPO = "aoxo/text2asmr-segments"
CLAP_SR = 48_000
BASE_UNFUSED = "laion/clap-htsat-unfused"
FT_MODEL = "aoxo/clap-htsat-unfused-asmr"


def hf_token() -> str:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    return token


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("file_name") or not r.get("gemini_label") or not r.get("clap_label"):
            continue
        rows.append(r)
    return rows


def stratified_sample(rows: list[dict], limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["gemini_label"]].append(r)
    # Round-robin so reject + rare classes aren't drowned by breathing.
    keys = sorted(by.keys(), key=lambda k: (-len(by[k]), k))
    for k in keys:
        rng.shuffle(by[k])
    out, idx = [], {k: 0 for k in keys}
    while len(out) < limit and any(idx[k] < len(by[k]) for k in keys):
        for k in keys:
            if len(out) >= limit:
                break
            i = idx[k]
            if i < len(by[k]):
                out.append(by[k][i])
                idx[k] = i + 1
    rng.shuffle(out)
    return out


def _features(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out


class Tagger:
    def __init__(self, clap_id: str, device: str, batch: int) -> None:
        import torch
        from transformers import ClapModel, ClapProcessor
        from text2asmr.data.ontology import BY_KEY, MARGIN, all_probes

        self.torch = torch
        self.device = device
        self.batch = batch
        # Always use base processor (FT repo may miss text tokenizer quirks).
        self.processor = ClapProcessor.from_pretrained(BASE_UNFUSED)
        self.model = ClapModel.from_pretrained(clap_id).to(device).eval()
        texts, self.owners = all_probes()
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            self.text_emb = _features(
                self.model.get_text_features(**{k: v.to(device) for k, v in inputs.items()})
            )
        self.pos = [i for i, o in enumerate(self.owners) if o is not None]
        self.neg = [i for i, o in enumerate(self.owners) if o is None]
        self.margin = MARGIN
        self.by_key = BY_KEY

    def predict(self, waves: list[np.ndarray]) -> list[str]:
        torch = self.torch
        out: list[str] = []
        for i in range(0, len(waves), self.batch):
            chunk = waves[i : i + self.batch]
            inputs = self.processor(
                audios=chunk, sampling_rate=CLAP_SR, return_tensors="pt", padding=True
            )
            if "is_longer" in inputs:
                inputs["is_longer"] = inputs["is_longer"].new_zeros(inputs["is_longer"].shape)
            with torch.no_grad():
                emb = _features(
                    self.model.get_audio_features(
                        **{k: v.to(self.device) for k, v in inputs.items()}
                    )
                )
                scores = emb @ self.text_emb.T
            for row in scores:
                best_pos, pos_idx = row[self.pos].max(dim=0)
                best_neg = row[self.neg].max() if self.neg else row.new_tensor(-1e9)
                if (best_pos - best_neg).item() < self.margin:
                    out.append("reject")
                else:
                    out.append(self.owners[self.pos[int(pos_idx)]])
        return out


def decode_flac(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != CLAP_SR:
        # light resample via linear (clips are short)
        n = int(round(len(audio) * CLAP_SR / sr))
        if n <= 1:
            return np.zeros(CLAP_SR, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    # cap 10s like fine-tune
    max_n = CLAP_SR * 10
    if audio.size > max_n:
        audio = audio[:max_n]
    return audio


def metrics(gold: list[str], pred: list[str], name: str) -> dict:
    n = len(gold)
    exact = sum(g == p for g, p in zip(gold, pred))
    # collapse breathing family for a softer score
    def fam(x: str) -> str:
        if x.startswith("breathing"):
            return "breathing*"
        return x

    soft = sum(fam(g) == fam(p) for g, p in zip(gold, pred))
    # among non-reject gold
    non_rej = [(g, p) for g, p in zip(gold, pred) if g != "reject"]
    rej_gold = [(g, p) for g, p in zip(gold, pred) if g == "reject"]
    acc_pos = sum(g == p for g, p in non_rej) / max(1, len(non_rej))
    # reject recall/precision
    tp_r = sum(p == "reject" for g, p in rej_gold)
    fp_r = sum(p == "reject" for g, p in non_rej)
    rec_r = tp_r / max(1, len(rej_gold))
    prec_r = tp_r / max(1, tp_r + fp_r)
    per = Counter()
    per_n = Counter()
    for g, p in zip(gold, pred):
        per_n[g] += 1
        if g == p:
            per[g] += 1
    print(f"\n=== {name} vs Gemini (n={n}) ===")
    print(f"exact_agree={exact}/{n} ({100*exact/n:.1f}%)")
    print(f"breathing_family_soft={soft}/{n} ({100*soft/n:.1f}%)")
    print(f"non_reject_exact={100*acc_pos:.1f}% (n={len(non_rej)})")
    print(f"reject_recall={100*rec_r:.1f}%  reject_precision={100*prec_r:.1f}%")
    print("per-class exact (gemini class):")
    for k, m in sorted(per_n.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k:22s} {per[k]:3d}/{m:3d} ({100*per[k]/max(m,1):.0f}%)")
    return {
        "name": name,
        "n": n,
        "exact": exact / n,
        "soft": soft / n,
        "non_reject_exact": acc_pos,
        "reject_recall": rec_r,
        "reject_precision": prec_r,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retag", type=Path, default=HERE / "label_tool" / "gemini_retag.jsonl")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--cache", type=Path, default=Path.home() / "t2a" / "clap_vs_gemini_cache")
    ap.add_argument("--out", type=Path, default=HERE / "label_tool" / "clap_vs_gemini_compare.json")
    args = ap.parse_args()

    import torch
    from huggingface_hub import hf_hub_download

    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
        log("MPS unavailable; using CPU")

    rows = load_rows(args.retag)
    sample = stratified_sample(rows, args.limit, args.seed)
    log(f"sample={len(sample)} from {len(rows)} dual-labeled rows")
    log("gemini mix: " + str(Counter(r["gemini_label"] for r in sample).most_common(8)))

    args.cache.mkdir(parents=True, exist_ok=True)
    waves = []
    keep = []
    for i, r in enumerate(sample):
        fn = r["file_name"]
        try:
            p = Path(
                hf_hub_download(
                    SEGMENTS_REPO,
                    f"triggers/{fn}",
                    repo_type="dataset",
                    token=hf_token(),
                    cache_dir=str(args.cache / "hf"),
                )
            )
            wav = decode_flac(p)
            if wav.size < CLAP_SR // 4:
                continue
            waves.append(wav)
            keep.append(r)
        except Exception as exc:  # noqa: BLE001
            if i < 5:
                log(f"skip {fn}: {type(exc).__name__}: {exc}")
        if (i + 1) % 50 == 0:
            log(f"downloaded {i+1}/{len(sample)} kept={len(keep)}")
    log(f"audio ready={len(keep)}")
    gold = [r["gemini_label"] for r in keep]
    stored = [r["clap_label"] for r in keep]

    results = []
    results.append(metrics(gold, stored, "stored_original_CLAP"))

    preds: dict[str, list[str]] = {}
    for name, mid in [
        ("base_unfused", BASE_UNFUSED),
        ("finetuned_asmr", FT_MODEL),
    ]:
        log(f"loading {mid} on {args.device}")
        tagger = Tagger(mid, args.device, args.batch)
        t0 = time.time()
        pred = tagger.predict(waves)
        log(f"{name} inferred in {time.time()-t0:.1f}s")
        preds[name] = pred
        results.append(metrics(gold, pred, name))
        del tagger
        if args.device == "mps":
            torch.mps.empty_cache()

    ft_pred = preds["finetuned_asmr"]
    base_pred = preds["base_unfused"]
    fix = sum((s != g and f == g) for g, s, f in zip(gold, stored, ft_pred))
    break_ = sum((s == g and f != g) for g, s, f in zip(gold, stored, ft_pred))
    both_wrong = sum((s != g and f != g) for g, s, f in zip(gold, stored, ft_pred))
    print(f"\n=== FT vs stored original (gold=Gemini) ===")
    print(f"FT_fixes_original={fix}")
    print(f"FT_breaks_original={break_}")
    print(f"both_wrong={both_wrong}")
    print(f"base_unfused_exact={sum(b==g for b,g in zip(base_pred,gold))}/{len(gold)}")

    payload = {
        "n": len(gold),
        "results": results,
        "ft_fixes_original": fix,
        "ft_breaks_original": break_,
        "both_wrong": both_wrong,
        "base_unfused_exact": sum(b == g for b, g in zip(base_pred, gold)),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
