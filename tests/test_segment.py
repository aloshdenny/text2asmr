import json

import pytest

from text2asmr.data.segment import (
    MAX_SPEECH_S,
    MIN_TRIGGER_S,
    Span,
    load_alignment,
    split_alignment,
    summarise,
)


def word(w, start, end):
    return {"type": "word", "word": w, "start": start, "end": end}


def silence(start, end):
    return {"type": "silence", "start": start, "end": end}


def test_consecutive_words_merge_into_one_phrase():
    spans = split_alignment(
        [word("hello", 0.0, 0.4), word("there", 0.5, 1.2)], "1", intro_skip_s=0
    )
    assert len(spans) == 1
    assert spans[0].kind == "speech"
    assert spans[0].text == "hello there"
    assert spans[0].start == 0.0 and spans[0].end == 1.2


def test_long_gap_breaks_the_phrase():
    spans = split_alignment(
        [word("one", 0.0, 1.0), word("two", 3.0, 4.2)], "1", intro_skip_s=0
    )
    assert [s.text for s in spans if s.kind == "speech"] == ["one", "two"]


def test_short_phrases_are_dropped():
    # 0.3 s is below MIN_SPEECH_S and unusable as a training example.
    assert split_alignment([word("hi", 0.0, 0.3)], "1", intro_skip_s=0) == []


def test_loud_gap_becomes_a_trigger_candidate():
    spans = split_alignment([silence(0.0, 5.0)], "1", intro_skip_s=0)
    assert len(spans) == 1
    assert spans[0].kind == "trigger_candidate"
    assert spans[0].duration == 5.0


def test_short_silence_is_not_a_trigger_candidate():
    assert split_alignment([silence(0.0, MIN_TRIGGER_S - 0.1)], "1", intro_skip_s=0) == []


def test_long_silence_is_chunked_into_windows():
    # A 60 s brushing pause is many examples, not one oversized one.
    spans = split_alignment([silence(0.0, 60.0)], "1", intro_skip_s=0)
    assert len(spans) == 5
    assert all(s.kind == "trigger_candidate" for s in spans)
    assert all(s.duration <= 12.0 for s in spans)
    assert spans[0].start == 0.0 and spans[-1].end == 60.0


def test_silence_remainder_below_minimum_is_dropped():
    # 13 s -> one 12 s window, then a 1 s remainder that is too short.
    spans = split_alignment([silence(0.0, 13.0)], "1", intro_skip_s=0)
    assert [s.duration for s in spans] == [12.0]


def test_overlong_word_run_is_split_at_max_speech():
    entries = [word(f"w{i}", i * 1.0, i * 1.0 + 0.9) for i in range(30)]
    spans = [s for s in split_alignment(entries, "1", intro_skip_s=0) if s.kind == "speech"]
    assert len(spans) > 1
    assert all(s.duration <= MAX_SPEECH_S for s in spans)


def test_speech_and_triggers_interleave_in_time_order():
    spans = split_alignment(
        [word("a", 0.0, 1.2), silence(1.2, 6.0), word("b", 6.0, 7.5)], "1", intro_skip_s=0
    )
    assert [s.kind for s in spans] == ["speech", "trigger_candidate", "speech"]
    assert [s.start for s in spans] == sorted(s.start for s in spans)


def test_entries_are_sorted_before_splitting():
    spans = split_alignment(
        [word("second", 2.0, 3.2), word("first", 0.0, 1.2)], "1", intro_skip_s=0
    )
    assert spans[0].text == "first" or "first" in spans[0].text


def test_malformed_entries_are_skipped():
    entries = [
        {"type": "word", "word": "x"},            # no timing
        {"type": "word", "word": "y", "start": 5.0, "end": 1.0},  # inverted
        word("good", 0.0, 1.5),
    ]
    spans = split_alignment(entries, "1", intro_skip_s=0)
    assert [s.text for s in spans] == ["good"]


def test_uid_is_stable_and_unique_per_span():
    spans = split_alignment([silence(0.0, 30.0)], "7", intro_skip_s=0)
    uids = [s.uid for s in spans]
    assert len(set(uids)) == len(uids)
    assert all(u.startswith("7_") for u in uids)


def test_load_alignment_accepts_a_bare_list(tmp_path):
    p = tmp_path / "1.json"
    p.write_text(json.dumps([word("hi", 0.0, 1.5)]))
    assert load_alignment(p)[0]["word"] == "hi"


def test_load_alignment_unwraps_a_wrapped_object(tmp_path):
    p = tmp_path / "1.json"
    p.write_text(json.dumps({"segments": [word("hi", 0.0, 1.5)]}))
    assert load_alignment(p)[0]["word"] == "hi"


def test_load_alignment_rejects_unrecognised_shape(tmp_path):
    p = tmp_path / "1.json"
    p.write_text(json.dumps({"nope": 1}))
    with pytest.raises(ValueError):
        load_alignment(p)


def test_summarise_reports_both_streams():
    spans = split_alignment(
        [word("a", 0.0, 1.2), silence(1.2, 20.0), word("b", 20.0, 21.5)], "1", intro_skip_s=0
    )
    out = summarise(spans)
    assert out["speech_segments"] == 2
    assert out["trigger_candidates"] >= 1
    assert out["words"] == 2
    assert out["speech_seconds"] == pytest.approx(2.7)
    assert out["trigger_seconds"] > 0


def test_short_inter_word_silences_do_not_break_the_phrase():
    # The aligner emits a silence entry between every pair of words. If those
    # broke phrases, every speech segment would be a single word.
    entries = [
        word("the", 0.0, 0.30), silence(0.30, 0.34),
        word("soft", 0.34, 0.70), silence(0.70, 0.75),
        word("brush", 0.75, 1.40),
    ]
    spans = split_alignment(entries, "1", intro_skip_s=0)
    assert len(spans) == 1
    assert spans[0].text == "the soft brush"


def test_long_silence_still_breaks_the_phrase_and_yields_a_trigger():
    entries = [
        word("one", 0.0, 1.2), silence(1.2, 9.0), word("two", 9.0, 10.4),
    ]
    kinds = [s.kind for s in split_alignment(entries, "1", intro_skip_s=0)]
    assert kinds == ["speech", "trigger_candidate", "speech"]
