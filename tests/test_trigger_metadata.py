import json

import pytest

from t2a.data import trigger_metadata as tm


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    # The module caches on first use; each test needs its own table.
    tm._CAPTIONS = None
    meta = tmp_path / "metadata.jsonl"
    meta.write_text("\n".join(json.dumps(r) for r in [
        {"file_name": "1_000001859.flac", "trigger": "blowing",
         "intensity": "mild", "caption": "ASMR mild blowing, close-mic binaural, no speech"},
        {"file_name": "1_000006489.flac", "trigger": "brushing",
         "intensity": "soft", "caption": "ASMR soft brushing, close-mic binaural, no speech"},
        "not json",
    ][:2]) + "\nnot json\n")
    monkeypatch.setenv("T2A_TRIGGER_METADATA", str(meta))
    yield
    tm._CAPTIONS = None


def test_caption_is_joined_on_file_name():
    out = tm.get_custom_metadata({"relpath": "1_000001859.flac"}, None)
    assert out["prompt"] == "ASMR mild blowing, close-mic binaural, no speech"
    assert out["trigger"] == "blowing"
    assert out["intensity"] == "mild"


def test_intensity_is_preserved_in_the_prompt():
    # [mild] and [soft] must not collapse to the same conditioning.
    a = tm.get_custom_metadata({"relpath": "1_000001859.flac"}, None)["prompt"]
    b = tm.get_custom_metadata({"relpath": "1_000006489.flac"}, None)["prompt"]
    assert "mild" in a and "soft" in b and a != b


def test_nested_relpath_still_matches():
    # Unpacked shards can leave a subdirectory in the path.
    out = tm.get_custom_metadata({"relpath": "shard-00000/1_000006489.flac"}, None)
    assert "brushing" in out["prompt"]


def test_path_key_used_when_relpath_absent():
    out = tm.get_custom_metadata({"path": "/data/1_000001859.flac"}, None)
    assert "blowing" in out["prompt"]


def test_unknown_file_gets_a_usable_fallback():
    out = tm.get_custom_metadata({"relpath": "nope.flac"}, None)
    assert out["prompt"] == tm._FALLBACK
    assert out["prompt"].strip()  # never an empty prompt


def test_torn_metadata_lines_are_skipped():
    # The builder can leave a torn final line; it must not break loading.
    assert tm.get_custom_metadata({"relpath": "1_000001859.flac"}, None)["prompt"]


def test_missing_metadata_file_falls_back(monkeypatch, tmp_path):
    tm._CAPTIONS = None
    monkeypatch.setenv("T2A_TRIGGER_METADATA", str(tmp_path / "absent.jsonl"))
    assert tm.get_custom_metadata({"relpath": "x.flac"}, None)["prompt"] == tm._FALLBACK
