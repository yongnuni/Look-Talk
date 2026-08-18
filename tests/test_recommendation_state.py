"""입력 문자열에 따른 favorites/autocomplete 표시 상태."""

import pytest

from src.recommendation.input_state import (
    AUTOCOMPLETE_MODE,
    FAVORITES_MODE,
    SuggestionStateController,
    format_suggestion_update,
)


def test_empty_input_has_blank_slots_and_first_jamo_switches_mode():
    calls = []

    def provider(prefix, limit):
        calls.append((prefix, limit))
        return ["감사합니다", "감사"]

    state = SuggestionStateController(suggestion_provider=provider)

    initial = state.update("")
    assert initial.mode == FAVORITES_MODE
    assert initial.items == ()
    assert initial.slots == ("", "", "")

    changed = state.update("ㄱ")
    assert changed.mode == AUTOCOMPLETE_MODE
    assert changed.chosung_input == "ㄱ"
    assert changed.slots == ("감사합니다", "감사", "")
    assert calls == [("ㄱ", 3)]


def test_full_deletion_returns_to_blank_favorite_slots():
    state = SuggestionStateController(
        suggestion_provider=lambda prefix, limit: ["한국"],
    )
    state.update("")
    state.update("ㅎ")

    changed = state.update("")

    assert changed.mode == FAVORITES_MODE
    assert changed.items == ()
    assert changed.slots == ("", "", "")
    assert state.autocomplete_suggestions == ()


def test_hospital_candidate_is_not_queried_or_shown_for_empty_input():
    calls = []

    def provider(prefix, limit):
        calls.append((prefix, limit))
        return ["도와주세요"]

    state = SuggestionStateController(suggestion_provider=provider)

    initial = state.update("")

    assert calls == []
    assert initial.mode == FAVORITES_MODE
    assert initial.items == ()
    assert initial.slots == ("", "", "")


@pytest.mark.parametrize("current_text", ["ㅏ", "ㆍ", "abc", "123"])
def test_nonempty_input_without_chosung_keeps_empty_autocomplete_mode(current_text):
    state = SuggestionStateController(
        suggestion_provider=lambda prefix, limit: pytest.fail("must not query"),
    )

    changed = state.update(current_text)

    assert changed.mode == AUTOCOMPLETE_MODE
    assert changed.chosung_input == ""
    assert changed.slots == ("", "", "")


def test_missing_candidate_does_not_restore_favorites_or_previous_results():
    responses = {"ㄱ": ["감사"], "ㅃ": []}
    state = SuggestionStateController(
        suggestion_provider=lambda prefix, limit: responses[prefix],
    )
    state.update("ㄱ")

    changed = state.update("ㅃ")

    assert changed.mode == AUTOCOMPLETE_MODE
    assert changed.items == ()
    assert changed.slots == ("", "", "")


def test_only_current_word_after_last_space_is_queried():
    calls = []

    def provider(prefix, limit):
        calls.append(prefix)
        return ["한국"]

    state = SuggestionStateController(suggestion_provider=provider)

    changed = state.update("안녕 ㅎㄱ")
    trailing_space = state.update("안녕 ㅎㄱ ")

    assert changed.chosung_input == "ㅎㄱ"
    assert calls == ["ㅎㄱ"]
    assert trailing_space.mode == AUTOCOMPLETE_MODE
    assert trailing_space.slots == ("", "", "")


def test_unchanged_text_is_not_queried_again():
    calls = []

    def provider(prefix, limit):
        calls.append(prefix)
        return ["감사"]

    state = SuggestionStateController(suggestion_provider=provider)

    assert state.update("ㄱ") is not None
    assert state.update("ㄱ") is None
    assert calls == ["ㄱ"]


def test_text_change_requeries_but_does_not_emit_identical_display_update():
    calls = []

    def provider(prefix, limit):
        calls.append(prefix)
        return ["감사"]

    state = SuggestionStateController(suggestion_provider=provider)
    state.update("ㄱ")

    assert state.update("가") is None
    assert calls == ["ㄱ", "ㄱ"]


def test_injected_favorites_are_kept_separate_and_padded():
    state = SuggestionStateController(
        favorite_sentences=["사용자 문장"],
        suggestion_provider=lambda prefix, limit: ["자동완성"],
    )

    initial = state.update("")
    active = state.update("ㅈ")

    assert initial.slots == ("사용자 문장", "", "")
    assert active.slots == ("자동완성", "", "")
    assert state.favorite_sentences == ("사용자 문장",)
    assert state.autocomplete_suggestions == ("자동완성",)


def test_query_error_warns_and_keeps_empty_autocomplete_slots():
    def provider(prefix, limit):
        raise RuntimeError("broken")

    state = SuggestionStateController(suggestion_provider=provider)

    with pytest.warns(RuntimeWarning, match="query failed"):
        changed = state.update("ㄱ")

    assert changed.mode == AUTOCOMPLETE_MODE
    assert changed.slots == ("", "", "")


def test_terminal_update_format_contains_mode_input_items_and_elapsed_time():
    state = SuggestionStateController(
        suggestion_provider=lambda prefix, limit: ["감사합니다"],
    )
    update = state.update("ㄱㅅ")

    message = format_suggestion_update(update)

    assert message.startswith("[suggestions] mode=autocomplete")
    assert "input='ㄱㅅ'" in message
    assert "items=['감사합니다']" in message
    assert "elapsed_ms=" in message
