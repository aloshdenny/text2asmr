import json
import sys
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text2asmr.native.dataset import SR, T2ANativeDataset, read_metadata


class FakeTokenizer:
    """Just enough of BertTokenizer's interface for the dataset to use."""

    def __call__(self, text, truncation, max_length, padding, return_tensors):
        import torch

        ids = torch.zeros(1, max_length, dtype=torch.long)
        mask = torch.ones(1, max_length, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}


def make_speech_dir(tmp_path: Path, n: int = 3, duration_s: float = 1.0) -> Path:
    d = tmp_path / "speech"
    d.mkdir()
    with (d / "metadata.jsonl").open("w") as f:
        for i in range(n):
            name = f"clip{i}.flac"
            audio = np.zeros(int(duration_s * SR), dtype=np.float32)
            sf.write(d / name, audio, SR)
            f.write(json.dumps({"file_name": name, "text": f"hello {i}"}) + "\n")
    return d


def make_triggers_dir(tmp_path: Path, n: int = 2) -> Path:
    d = tmp_path / "triggers"
    d.mkdir()
    with (d / "metadata.jsonl").open("w") as f:
        for i in range(n):
            name = f"trig{i}.flac"
            audio = np.zeros((int(0.5 * SR), 2), dtype=np.float32)  # stereo
            sf.write(d / name, audio, SR)
            f.write(json.dumps({
                "file_name": name, "caption": f"ASMR tapping {i}",
            }) + "\n")
    return d


def test_read_metadata_skips_torn_lines(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{bad json\n')
    rows = read_metadata(p)
    assert rows == [{"a": 1}, {"a": 2}]


def test_read_metadata_missing_file_returns_empty(tmp_path):
    assert read_metadata(tmp_path / "absent.jsonl") == []


def test_dataset_combines_speech_and_triggers(tmp_path):
    speech = make_speech_dir(tmp_path, n=3)
    triggers = make_triggers_dir(tmp_path, n=2)
    ds = T2ANativeDataset(speech, triggers, window_s=0.5,
                          length_multiple=16, tokenizer=FakeTokenizer())
    assert len(ds) == 5
    kinds = {ds.items[i][2] for i in range(len(ds))}
    assert kinds == {"speech", "trigger"}


def test_window_length_is_multiple_of_length_multiple(tmp_path):
    speech = make_speech_dir(tmp_path, n=1)
    ds = T2ANativeDataset(speech, None, window_s=0.37,
                          length_multiple=16, tokenizer=FakeTokenizer())
    assert ds.window % 16 == 0


def test_short_audio_is_padded_to_window(tmp_path):
    speech = make_speech_dir(tmp_path, n=1, duration_s=0.1)
    ds = T2ANativeDataset(speech, None, window_s=1.0,
                          length_multiple=16, tokenizer=FakeTokenizer())
    item = ds[0]
    assert item["waveform"].shape[-1] == ds.window


def test_long_audio_is_cropped_to_window(tmp_path):
    speech = make_speech_dir(tmp_path, n=1, duration_s=3.0)
    ds = T2ANativeDataset(speech, None, window_s=0.5,
                          length_multiple=16, tokenizer=FakeTokenizer())
    item = ds[0]
    assert item["waveform"].shape[-1] == ds.window


def test_stereo_trigger_audio_becomes_mono(tmp_path):
    triggers = make_triggers_dir(tmp_path, n=1)
    ds = T2ANativeDataset(None, triggers, window_s=0.3,
                          length_multiple=16, tokenizer=FakeTokenizer())
    item = ds[0]
    assert item["waveform"].shape[0] == 1  # channel dim, not stereo


def test_trigger_without_caption_or_tag_is_skipped(tmp_path):
    d = tmp_path / "triggers"
    d.mkdir()
    audio = np.zeros(int(0.5 * SR), dtype=np.float32)
    sf.write(d / "x.flac", audio, SR)
    (d / "metadata.jsonl").write_text(
        json.dumps({"file_name": "x.flac"}) + "\n"  # no caption, no tag
    )
    with pytest.raises(ValueError):
        T2ANativeDataset(None, d, window_s=0.3, length_multiple=16,
                         tokenizer=FakeTokenizer())


def test_raises_when_both_dirs_empty(tmp_path):
    with pytest.raises(ValueError):
        T2ANativeDataset(None, None, window_s=0.5, length_multiple=16,
                         tokenizer=FakeTokenizer())


def test_missing_audio_file_is_silently_excluded(tmp_path):
    d = tmp_path / "speech"
    d.mkdir()
    (d / "metadata.jsonl").write_text(
        json.dumps({"file_name": "ghost.flac", "text": "hi"}) + "\n"
    )
    with pytest.raises(ValueError):
        T2ANativeDataset(d, None, window_s=0.5, length_multiple=16,
                         tokenizer=FakeTokenizer())
