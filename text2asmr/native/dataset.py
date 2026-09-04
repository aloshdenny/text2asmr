"""Unified speech+trigger dataset for the native T2A model.

The paper's architecture generates both speech and non-speech ASMR elements
from one model (the abstract: "produce ... audio containing both speech and
non-speech elements"), unlike the split Chatterbox/Stable-Audio approach.
This dataset mixes both corpora so one model trains on both: speech rows
supply their transcript as text, trigger rows supply their bracket-tag
caption (``[soft][brushing]``-style, matching grammar.py), and audio from
either source is resampled to one common rate.

22.05 kHz mono matches the paper's own choice (Section IV.A), not an
arbitrary pick -- reproducing the paper's architecture on a different sample
rate would be a different experiment.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SR = 22050


def read_metadata(meta_path: Path) -> list[dict]:
    """Defensive JSONL read: torn last lines and bad rows are dropped, not fatal.

    Every prior dataset in this project has needed this same tolerance --
    interrupted writes leave a torn final line often enough that skipping it
    silently is the right default, not an edge case.
    """
    rows = []
    if not meta_path.exists():
        return rows
    for line in meta_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


class T2ANativeDataset(Dataset):
    def __init__(self, speech_dir: Path | None, triggers_dir: Path | None,
                window_s: float, length_multiple: int, tokenizer,
                max_text_len: int = 32) -> None:
        self.window = int(window_s * SR)
        # DownBlocks halve the length each stage; padding to a multiple keeps
        # every skip connection's shape aligned with its matching up block.
        self.window -= self.window % length_multiple
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        self.items: list[tuple[Path, str, str]] = []  # (audio_path, text, kind)
        if speech_dir is not None:
            for r in read_metadata(speech_dir / "metadata.jsonl"):
                p = speech_dir / r["file_name"]
                if p.exists() and r.get("text"):
                    self.items.append((p, r["text"], "speech"))
        if triggers_dir is not None:
            for r in read_metadata(triggers_dir / "metadata.jsonl"):
                p = triggers_dir / r["file_name"]
                text = r.get("caption") or r.get("tag")
                if p.exists() and text:
                    self.items.append((p, text, "trigger"))

        if not self.items:
            raise ValueError(f"no usable rows under {speech_dir} or {triggers_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, path: Path) -> np.ndarray:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)  # mono, regardless of source channel count
        if sr != SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        return audio

    def _fit_window(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        if n >= self.window:
            start = random.randint(0, n - self.window)
            return audio[start:start + self.window]
        pad = self.window - n
        return np.pad(audio, (0, pad))

    def __getitem__(self, idx: int) -> dict:
        path, text, kind = self.items[idx]
        try:
            audio = self._fit_window(self._load_audio(path))
        except Exception:  # noqa: BLE001 - one bad file must not kill a batch
            audio = np.zeros(self.window, dtype=np.float32)

        enc = self.tokenizer(text, truncation=True, max_length=self.max_text_len,
                             padding="max_length", return_tensors="pt")
        return {
            "waveform": torch.from_numpy(audio).float().unsqueeze(0),  # (1, T)
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "kind": kind,
        }
