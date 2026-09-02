#!/usr/bin/env python3
"""Build both training sets from `aoxo/audios`, tagging triggers with CLAP.

Runs on the GPU box. For each source file it streams the m4a, cuts it using the
word-level alignment, and writes two streams:

  speech/   24 kHz mono FLAC + transcript   -> TTS finetune
  triggers/ 48 kHz stereo FLAC + caption    -> audio-diffusion finetune

Disk discipline matters: 158 h of speech at 48 kHz stereo would be ~109 GB, so
speech is written at the rate the TTS backend actually trains on, and triggers
are capped per class by ``--trigger-hours`` and kept at native rate because the
audio model needs the stereo image and the top octave.

Each source file is deleted from the HF cache once processed, and progress is
checkpointed to ``state.json``, so the job resumes after an interruption.

    python scripts/build_datasets.py --out ~/t2a/out --limit 50   # smoke test
    python scripts/build_datasets.py --out ~/t2a/out              # full run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t2a.data.ontology import (  # noqa: E402
    MARGIN,
    all_probes,
    intensity_from_loudness,
)
from t2a.data.segment import load_alignment, split_alignment  # noqa: E402

REPO = "aoxo/audios"
CLAP = "laion/clap-htsat-fused"
CLAP_SR = 48_000
SPEECH_SR = 24_000
TRIGGER_SR = 48_000

# A gap whose loudness sits at the noise floor is real silence, not a trigger.
# -50 dBFS is below any deliberate ASMR gesture but above dithered digital
# silence, so it separates room tone from quiet brushing.
SILENCE_FLOOR_DB = -50.0


def decode(path: Path, sr: int, mono: bool, start: float, dur: float) -> np.ndarray:
    """Decode one span via ffmpeg. Returns (channels, samples) float32.

    ffmpeg is used rather than librosa because seeking directly to the span
    avoids decoding the whole 13-minute file once per segment.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr), "-ac", "1" if mono else "2", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(out, dtype=np.float32)
    if mono:
        return audio.reshape(1, -1)
    return audio.reshape(-1, 2).T if audio.size else audio.reshape(2, 0)


def rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-9))


def _features(out):
    """Extract CLAP's projected embedding from a get_*_features return value.

    transformers 5.x returns a BaseModelOutputWithPooling whose ``pooler_output``
    is the projection-dim embedding, already L2-normalised by the model. Older
    versions returned that tensor directly, so accept both and do not
    re-normalise.
    """
    return out.pooler_output if hasattr(out, "pooler_output") else out


class Tagger:
    """Zero-shot CLAP tagging against the ASMR ontology."""

    def __init__(self, device: str, batch: int) -> None:
        import torch
        from transformers import ClapModel, ClapProcessor

        self.torch = torch
        self.device = device
        self.batch = batch
        self.model = ClapModel.from_pretrained(CLAP).to(device).eval()
        self.processor = ClapProcessor.from_pretrained(CLAP)

        texts, self.owners = all_probes()
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            self.text_emb = _features(
                self.model.get_text_features(
                    **{k: v.to(device) for k, v in inputs.items()}
                )
            )
        # Index once, so scoring is a single matmul plus two segment-maxes.
        self.pos = [i for i, o in enumerate(self.owners) if o is not None]
        self.neg = [i for i, o in enumerate(self.owners) if o is None]

    def tag(self, clips: list[np.ndarray]) -> list[tuple[str, float] | None]:
        """Return (trigger_key, confidence) per clip, or None if rejected."""
        torch = self.torch
        results: list[tuple[str, float] | None] = []
        for i in range(0, len(clips), self.batch):
            chunk = clips[i : i + self.batch]
            inputs = self.processor(
                audio=chunk, sampling_rate=CLAP_SR, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                emb = _features(self.model.get_audio_features(
                    **{k: v.to(self.device) for k, v in inputs.items()}
                ))
                scores = emb @ self.text_emb.T

            for row in scores:
                best_pos, pos_idx = row[self.pos].max(dim=0)
                best_neg = row[self.neg].max()
                # Reject anything a negative probe explains as well: those are
                # speech bleed, room tone or music, not triggers.
                if (best_pos - best_neg).item() < MARGIN:
                    results.append(None)
                else:
                    key = self.owners[self.pos[int(pos_idx)]]
                    results.append((key, float(best_pos)))
        return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all paired files")
    ap.add_argument("--trigger-hours", type=float, default=25.0)
    ap.add_argument("--speech-hours", type=float, default=120.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep-cache", action="store_true",
                    help="do not delete source audio after processing")
    args = ap.parse_args()

    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    out = args.out
    (out / "speech").mkdir(parents=True, exist_ok=True)
    (out / "triggers").mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "done": [], "speech_s": 0.0, "trigger_s": 0.0, "per_class": {}
    }
    done = set(state["done"])

    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    ids = sorted({f[:-5] for f in files if f.endswith(".json")}
                 & {f[:-4] for f in files if f.endswith(".m4a")},
                 key=lambda x: (len(x), x))
    if args.limit:
        ids = ids[: args.limit]
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} paired files, {len(done)} already done, {len(todo)} to go",
          flush=True)

    tagger = Tagger(args.device, args.batch)
    print("CLAP loaded", flush=True)

    # Cap per class so one loud, common trigger cannot dominate the set.
    n_classes = len({o for o in tagger.owners if o})
    per_class_cap = args.trigger_hours * 3600 / max(n_classes, 1) * 2.5

    speech_meta = (out / "speech" / "metadata.jsonl").open("a")
    trig_meta = (out / "triggers" / "metadata.jsonl").open("a")
    t0 = time.time()

    for n, fid in enumerate(todo, 1):
        if state["speech_s"] >= args.speech_hours * 3600 and \
           state["trigger_s"] >= args.trigger_hours * 3600:
            print("both quotas reached", flush=True)
            break
        try:
            jp = Path(hf_hub_download(REPO, f"{fid}.json", repo_type="dataset"))
            ap_ = Path(hf_hub_download(REPO, f"{fid}.m4a", repo_type="dataset"))
            spans = split_alignment(load_alignment(jp), fid)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{fid}] skipped: {type(exc).__name__}: {exc}", flush=True)
            state["done"].append(fid)
            continue

        # --- speech ---
        if state["speech_s"] < args.speech_hours * 3600:
            for span in (s for s in spans if s.kind == "speech"):
                try:
                    audio = decode(ap_, SPEECH_SR, True, span.start, span.duration)
                except subprocess.CalledProcessError:
                    continue
                if audio.size == 0 or rms_db(audio) < SILENCE_FLOOR_DB:
                    continue
                rel = f"{span.uid}.flac"
                sf.write(out / "speech" / rel, audio[0], SPEECH_SR)
                speech_meta.write(json.dumps({
                    "file_name": rel, "text": span.text, "source": fid,
                    "start": round(span.start, 3), "duration": round(span.duration, 3),
                }) + "\n")
                state["speech_s"] += span.duration

        # --- triggers ---
        cands = [s for s in spans if s.kind == "trigger_candidate"]
        if cands and state["trigger_s"] < args.trigger_hours * 3600:
            clips, kept = [], []
            for span in cands:
                try:
                    mono = decode(ap_, CLAP_SR, True, span.start, span.duration)
                except subprocess.CalledProcessError:
                    continue
                if mono.size == 0:
                    continue
                level = rms_db(mono)
                if level < SILENCE_FLOOR_DB:
                    continue  # true room tone, not a trigger
                clips.append(mono[0])
                kept.append((span, level))

            for (span, level), tagged in zip(kept, tagger.tag(clips)):
                if tagged is None:
                    continue
                key, conf = tagged
                used = state["per_class"].get(key, 0.0)
                if used >= per_class_cap:
                    continue
                try:
                    stereo = decode(ap_, TRIGGER_SR, False, span.start, span.duration)
                except subprocess.CalledProcessError:
                    continue
                if stereo.size == 0:
                    continue
                modifier, scaled = intensity_from_loudness(level)
                rel = f"{span.uid}.flac"
                sf.write(out / "triggers" / rel, stereo.T, TRIGGER_SR)
                trig_meta.write(json.dumps({
                    "file_name": rel,
                    "trigger": key,
                    "intensity": modifier,
                    "intensity_scaled": round(scaled, 3),
                    "tag": f"[{modifier}][{key}]",
                    "caption": f"ASMR {modifier} {key}, close-mic binaural, no speech",
                    "clap_confidence": round(conf, 4),
                    "rms_db": round(level, 2),
                    "source": fid,
                    "start": round(span.start, 3),
                    "duration": round(span.duration, 3),
                    "channels": int(stereo.shape[0]),
                }) + "\n")
                state["trigger_s"] += span.duration
                state["per_class"][key] = used + span.duration

        if not args.keep_cache:
            ap_.unlink(missing_ok=True)

        state["done"].append(fid)
        speech_meta.flush(); trig_meta.flush()
        state_path.write_text(json.dumps(state))

        if n % 5 == 0 or n == len(todo):
            rate = n / max(time.time() - t0, 1e-6) * 3600
            print(f"[{n}/{len(todo)}] speech {state['speech_s']/3600:.1f}h  "
                  f"triggers {state['trigger_s']/3600:.1f}h  "
                  f"{rate:.0f} files/h", flush=True)

    speech_meta.close(); trig_meta.close()
    print("\nper-class trigger seconds:")
    for k, v in sorted(state["per_class"].items(), key=lambda kv: -kv[1]):
        print(f"  {v/3600:6.2f} h  {k}")
    print(f"\nspeech {state['speech_s']/3600:.1f} h, "
          f"triggers {state['trigger_s']/3600:.1f} h -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
