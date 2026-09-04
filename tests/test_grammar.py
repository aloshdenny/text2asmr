from text2asmr.compose.grammar import (
    DEFAULT_INTENSITY,
    INTENSITIES,
    Speech,
    Trigger,
    parse,
    survey_vocabulary,
)


def test_plain_speech_is_one_segment():
    script = parse("hello there, welcome back")
    assert script.segments == [Speech("hello there, welcome back")]
    assert script.is_pure_speech


def test_bare_trigger_gets_default_intensity():
    script = parse("[tapping]")
    (trigger,) = script.segments
    assert isinstance(trigger, Trigger)
    assert trigger.name == "tapping"
    assert trigger.intensity == DEFAULT_INTENSITY
    assert trigger.modifier is None
    assert script.is_pure_trigger


def test_intensity_binds_to_following_trigger():
    (trigger,) = parse("[soft][brushing]").segments
    assert trigger.name == "brushing"
    assert trigger.modifier == "soft"
    assert trigger.intensity == INTENSITIES["soft"]


def test_interleaved_speech_and_triggers_keep_order():
    script = parse("relax now [vigorous][crinkling] and breathe")
    assert [s.kind for s in script.segments] == ["speech", "trigger", "speech"]
    assert script.segments[0].text == "relax now"
    assert script.segments[1].name == "crinkling"
    assert script.segments[1].intensity == INTENSITIES["vigorous"]
    assert script.segments[2].text == "and breathe"
    assert not script.is_pure_speech and not script.is_pure_trigger


def test_dangling_intensity_before_speech_is_dropped():
    # "[soft]" modifies nothing here, so it must not leak into the render plan.
    script = parse("[soft] just breathe")
    assert script.segments == [Speech("just breathe")]


def test_dangling_intensity_at_end_is_dropped():
    script = parse("[tapping][loud]")
    assert len(script.segments) == 1
    assert script.segments[0].name == "tapping"


def test_intensity_does_not_carry_across_a_trigger():
    a, b = parse("[soft][brushing][tapping]").segments
    assert a.modifier == "soft"
    assert b.modifier is None
    assert b.intensity == DEFAULT_INTENSITY


def test_tags_are_case_and_space_insensitive():
    (trigger,) = parse("[ Soft ][ BRUSHING ]").segments
    assert trigger.name == "brushing"
    assert trigger.modifier == "soft"


def test_whitespace_from_tag_removal_is_collapsed():
    script = parse("okay   [tapping]   so    then")
    assert script.segments[0].text == "okay"
    assert script.segments[2].text == "so then"


def test_trigger_prompt_expands_tags_for_the_audio_model():
    prompt = parse("[soft][brushing]").segments[0].prompt
    assert "[" not in prompt
    assert "soft brushing" in prompt


def test_survey_separates_open_triggers_from_closed_intensities():
    triggers, intensities = survey_vocabulary(
        ["[soft][brushing] hi", "[brushing]", "[mild][page turning]"]
    )
    assert triggers == {"brushing": 2, "page turning": 1}
    assert intensities == {"soft": 1, "mild": 1}


def test_survey_tolerates_empty_and_none_rows():
    triggers, intensities = survey_vocabulary(["", None, "[tapping]"])
    assert triggers == {"tapping": 1}
    assert intensities == {}
