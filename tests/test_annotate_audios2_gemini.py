from scripts.annotate_audios2_gemini import context_from_entries, paired_sources


def test_context_takes_two_words_before_and_after_the_gap():
    entries = [
        {"type": "word", "word": "some", "start": 0.0, "end": 0.3},
        {"type": "word", "word": "tapping", "start": 0.4, "end": 0.9},
        {"type": "silence", "start": 0.9, "end": 4.0},
        {"type": "word", "word": "right", "start": 4.0, "end": 4.3},
        {"type": "word", "word": "there", "start": 4.4, "end": 4.8},
    ]
    before, after = context_from_entries(entries, 0.9, 3.1)
    assert before == "some tapping"
    assert after == "right there"


def test_paired_sources_require_m4a_plus_m4a_json():
    files = [
        "a/one.m4a",
        "a/one.m4a.json",
        "b/two.m4a",
        "c/three.json",
        "d/four.m4a",
        "d/four.json",
        "skip/me.txt",
    ]
    assert paired_sources(files, None) == ["a/one.m4a"]
    assert paired_sources(files, {"a"}) == ["a/one.m4a"]
    assert paired_sources(files, {"b"}) == []
