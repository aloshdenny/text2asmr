#!/usr/bin/env python3
"""Can the research 4090 replace the RunPod labeller? Qwen3-Omni-30B-A3B, 4-bit, plain transformers (no vLLM
on native Windows), timed on real queued clips with the production prompt. Prints clips/hour and agreement
with the vLLM labels already in the ledger for the same clips, so speed and quality are both measured.

  python bench_qwen_local.py --n 64 --batch 8
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_audios2_qwen3 import PROMPT

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SR = 16_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--work", type=Path, default=Path(r"D:\t2a\qbench"))
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import BitsAndBytesConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    a.work.mkdir(parents=True, exist_ok=True)

    # clips Qwen-vLLM already labelled, so the local model's answers can be compared like for like
    labs = {}
    for l in open(hf_hub_download("aoxo/t2a-daddy", "labels/qwen3omni.jsonl", repo_type="dataset",
                                  local_dir=str(a.work / "lab")), encoding="utf-8"):
        r = json.loads(l); labs[r["uid"]] = r["label"]
        if len(labs) > 200000: break
    by_src = {}
    for u in labs: by_src.setdefault(u.rsplit("_", 1)[0], []).append(u)
    srcs = sorted(by_src, key=lambda s: -len(by_src[s]))[:3]
    clips = []
    for s in srcs:
        local = hf_hub_download("aoxo/t2a-daddy", s, repo_type="dataset", local_dir=str(a.work / "dl"))
        for u in by_src[s][: a.n // len(srcs) + 1]:
            t0 = int(u.rsplit("_", 1)[1]) / 1000 - 1.0
            pcm = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t0):.3f}", "-t", "8", "-i", local, "-f", "f32le",
                                  "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
            clips.append((u, np.frombuffer(pcm, np.float32).copy()))
        os.remove(local)
    clips = clips[: a.n]
    print(f"{len(clips)} clips from {len(srcs)} recordings", flush=True)

    t = time.time()
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, device_map="cuda", dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    if hasattr(model, "disable_talker"): model.disable_talker()
    print(f"model loaded in {time.time() - t:.0f} s, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    preds, t = [], time.time()
    for i in range(0, len(clips), a.batch):
        b = clips[i:i + a.batch]
        inp = proc(text=[text] * len(b), audio=[w for _, w in b], sampling_rate=SR, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=12, do_sample=False, return_audio=False, thinker_return_dict_in_generate=False)
        seq = out[0] if isinstance(out, tuple) else out
        ans = proc.batch_decode(seq[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        preds += [(u, s.strip().lower()) for (u, _), s in zip(b, ans)]
    el = time.time() - t
    agree = sum(labs[u] in p or p in labs[u] for u, p in preds) / max(1, len(preds))
    print(f"RESULT {len(preds)} clips in {el:.0f} s = {len(preds) / el * 3600:.0f} clips/h, "
          f"agreement with vLLM labels {agree:.0%}, peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    for u, p in preds[:6]: print(f"  {labs[u]:14} vs local: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
