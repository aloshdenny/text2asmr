import json
import sys
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import concat_speech as cs  # noqa: E402


def make_corpus(tmp_path: Path, segments: list[tuple[str, float, float, str]]):
    """segments: (source, start, duration, text). Each becomes a 1s@8kHz clip."""
    speech = tmp_path / "speech"
    speech.mkdir()
    meta = speech / "metadata.jsonl"
    sr = 8000
    with meta.open("w") as f:
        for source, start, dur, text in segments:
            name = f"{source}_{int(start*1000):09d}.flac"
            n = int(dur * sr)
            audio = np.full(n, 0.1, dtype=np.float32)
            sf.write(speech / name, audio, sr)
            f.write(json.dumps({
                "file_name": name, "text": text, "source": source,
                "start": start, "duration": dur,
            }) + "\n")
    return tmp_path


def run(tmp_path, target_s=5.0, max_s=8.0, min_segments=2):
    import argparse

    ns = argparse.Namespace(
        out=tmp_path, dest_name="speech_long", target_s=target_s, max_s=max_s,
        join_silence_ms=100, min_segments=min_segments,
    )
    sys.argv = ["concat_speech.py"]
    orig_parse = cs.argparse.ArgumentParser.parse_args
    cs.argparse.ArgumentParser.parse_args = lambda self: ns
    try:
        cs.main()
    finally:
        cs.argparse.ArgumentParser.parse_args = orig_parse
    meta = tmp_path / "speech_long" / "metadata.jsonl"
    return [json.loads(l) for l in meta.read_text().splitlines() if l.strip()]


def test_no_duplicate_groups_across_multiple_flushes(tmp_path):
    # Regression test: a mutable-default-argument flush() froze its `group`
    # reference at definition time, so every flush after the first rebind
    # re-wrote the same stale (empty, post-clear) group forever. Enough
    # segments to force several flushes for one source is what exposed it.
    segs = [("A", i * 1.0, 1.0, f"word{i}") for i in range(12)]
    make_corpus(tmp_path, segs)
    rows = run(tmp_path, target_s=3.0, max_s=4.0)
    names = [r["file_name"] for r in rows]
    assert len(names) == len(set(names)), "duplicate output rows"
    assert len(rows) >= 3  # 12 one-second segments at target 3s -> several groups


def test_concatenated_text_preserves_order(tmp_path):
    segs = [("A", 0.0, 1.0, "one"), ("A", 1.0, 1.0, "two"), ("A", 2.0, 1.0, "three")]
    make_corpus(tmp_path, segs)
    rows = run(tmp_path, target_s=10.0, max_s=10.0, min_segments=2)
    assert len(rows) == 1
    assert rows[0]["text"] == "one two three"
    assert rows[0]["n_segments"] == 3


def test_groups_split_at_max_s(tmp_path):
    segs = [("A", i * 3.0, 3.0, f"seg{i}") for i in range(4)]  # 4x3s = 12s
    make_corpus(tmp_path, segs)
    rows = run(tmp_path, target_s=100.0, max_s=7.0, min_segments=1)
    assert all(r["duration"] <= 7.5 for r in rows)  # allow join-silence slack
    assert sum(r["n_segments"] for r in rows) == 4


def test_lone_segment_group_dropped_by_default_min(tmp_path):
    segs = [("A", 0.0, 1.0, "only one")]
    make_corpus(tmp_path, segs)
    rows = run(tmp_path, target_s=5.0, max_s=5.0, min_segments=2)
    assert rows == []


def test_different_sources_never_mixed_in_one_group(tmp_path):
    segs = [("A", 0.0, 1.0, "a1"), ("B", 0.0, 1.0, "b1"), ("A", 1.0, 1.0, "a2")]
    make_corpus(tmp_path, segs)
    rows = run(tmp_path, target_s=10.0, max_s=10.0, min_segments=1)
    for r in rows:
        assert r["source"] in ("A", "B")
    texts = {r["text"] for r in rows}
    assert not any("a1" in t and "b1" in t for t in texts)


def test_output_audio_duration_matches_sum_of_parts_plus_silence(tmp_path):
    segs = [("A", 0.0, 1.0, "one"), ("A", 1.0, 1.0, "two")]
    make_corpus(tmp_path, segs)
    rows = run(tmp_path, target_s=10.0, max_s=10.0, min_segments=2,)
    r = rows[0]
    wav = tmp_path / "speech_long" / r["file_name"]
    info = sf.info(wav)
    assert info.duration == pytest.approx(2.0 + 0.1, abs=0.02)
