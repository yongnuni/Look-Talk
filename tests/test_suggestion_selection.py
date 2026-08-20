"""추천 후보의 hit-test, 선택 트리거, 입력 반영 및 로그 회귀 검증."""

import pytest

import src.hangul as hangul
from src.cheonjiin import cheonjiin_composer
from src.common import clock
from src.config import DWELL_SEC
from src.keyboard import (
    Button,
    KEYBOARD_LAYOUT_CHEONJIIN,
    KEYBOARD_LAYOUT_QWERTY,
    PendingWordBoundaryState,
    apply_suggestion,
    calculate_keyboard_layout,
    create_buttons,
    create_cheonjiin_buttons,
    keys_kor_normal,
    process_key,
)
from src.metrics.input_event_logger import InputEventLogger
from src.metrics.tap_logging import log_input_tap
from src.recommendation.input_state import SuggestionStateController
from src.recommendation.selection import (
    SuggestionTarget,
    hit_test_suggestion_target,
    resolve_input_target,
)
from src.tracking.dwell import DwellController
from src.tracking.mouth import MouthClickDetector


@pytest.fixture(autouse=True)
def _reset_input_state():
    hangul.finalText = ""
    hangul.jamo_buffer[:] = ['', '', '']
    cheonjiin_composer.reset()
    clock._reset()
    clock.init()
    yield
    hangul.finalText = ""
    hangul.jamo_buffer[:] = ['', '', '']
    cheonjiin_composer.reset()
    clock._reset()


def _layout_parts():
    layout = calculate_keyboard_layout(1280, 720)
    return (
        layout["suggestion_rects"],
        create_buttons(keys_kor_normal, layout),
    )


def _qwerty_input(prefix, jamos):
    hangul.finalText = prefix
    buttons = create_buttons(keys_kor_normal)
    is_korean, is_shift = True, False
    for jamo in jamos:
        is_korean, is_shift, buttons, _, _ = process_key(
            jamo,
            is_korean,
            is_shift,
            buttons,
            KEYBOARD_LAYOUT_QWERTY,
        )
    return hangul.finalText + hangul.compose_jamo_buffer()


def _current_text(keyboard_layout, is_korean=True):
    text = hangul.finalText + hangul.compose_jamo_buffer()
    if keyboard_layout == KEYBOARD_LAYOUT_CHEONJIIN and is_korean:
        text += cheonjiin_composer.get_pending_preview()
    return text


def _select_with_pending_boundary(text, keyboard_layout):
    boundary = PendingWordBoundaryState()
    apply_suggestion(text, True, keyboard_layout)
    boundary.mark_pending()
    return boundary


def test_nonempty_suggestion_hit_test_and_empty_slot_is_not_selectable():
    rects, _ = _layout_parts()
    slots = ("감사합니다", "", "도와주세요")

    target = hit_test_suggestion_target(rects, slots, *rects[0].center)

    assert target == SuggestionTarget(0, "감사합니다")
    assert hit_test_suggestion_target(
        rects,
        slots,
        *rects[1].center,
    ) is None


def test_leaving_suggestion_resets_dwell_progress():
    target = SuggestionTarget(0, "감사합니다")
    dwell = DwellController()

    dwell.update_target(target, now=1.0)
    _, ratio, _ = dwell.update_target(target, now=1.0 + DWELL_SEC / 2)
    assert ratio == pytest.approx(0.5)

    assert dwell.update_target(None, now=1.0 + DWELL_SEC / 2) == (
        None,
        0.0,
        None,
    )
    assert dwell.dwell_key is None
    assert dwell.dwell_start is None


def test_moving_to_another_suggestion_restarts_dwell():
    first = SuggestionTarget(0, "감사합니다")
    second = SuggestionTarget(1, "간호사 불러주세요")
    dwell = DwellController()

    dwell.update_target(first, now=1.0)
    dwell.update_target(first, now=1.0 + DWELL_SEC / 2)
    hovered, ratio, clicked = dwell.update_target(second, now=2.0)

    assert (hovered, ratio, clicked) == (second, 0.0, None)
    assert dwell.dwell_key == second
    assert dwell.dwell_start == 2.0


def test_suggestion_list_change_resets_same_target_progress_and_mouth_lock():
    target = SuggestionTarget(0, "감사합니다")
    dwell = DwellController()
    mouth = MouthClickDetector()
    original = ("감사합니다", "안녕하세요", "")
    changed = ("감사합니다", "도와주세요", "")

    dwell.set_target_context(original)
    mouth.set_target_context(original)
    dwell.update_target(target, now=1.0)
    dwell.update_target(target, now=1.0 + DWELL_SEC / 2)
    mouth.locked_key = target

    assert dwell.set_target_context(changed) is True
    assert mouth.set_target_context(changed) is True
    assert dwell.update_target(target, now=2.0) == (target, 0.0, None)
    assert mouth.locked_key is None


def test_dwell_completion_selects_once_then_cleared_slots_block_repeat():
    rects, buttons = _layout_parts()
    state = SuggestionStateController(
        suggestion_provider=lambda prefix, limit: ["감사합니다"],
    )
    state.update("ㄱㅅ")
    point = rects[0].center
    target = resolve_input_target(rects, state.slots, buttons, *point)
    dwell = DwellController()
    dwell.set_target_context(state.slots)

    dwell.update_target(target, now=1.0)
    _, _, clicked = dwell.update_target(
        target,
        now=1.0 + DWELL_SEC + 0.01,
    )
    assert clicked == SuggestionTarget(0, "감사합니다")

    hangul.finalText = "ㄱ"
    hangul.jamo_buffer[:] = ["ㅅ", "", ""]
    apply_suggestion(clicked.text, True, KEYBOARD_LAYOUT_QWERTY)
    state.clear_after_selection(hangul.finalText)
    dwell.set_target_context(state.slots)

    assert state.slots == ("", "", "")
    assert resolve_input_target(rects, state.slots, buttons, *point) is None
    assert dwell.update_target(None, now=3.0) == (None, 0.0, None)


def test_mouth_mode_selects_locked_suggestion_once(monkeypatch):
    target = SuggestionTarget(0, "감사합니다")
    detector = MouthClickDetector(
        lock_time=0.05,
        hold_time=0.20,
        cooldown=0.0,
    )
    mars = iter([0.0, 0.0, 0.5, 0.5, 0.5])
    times = iter([1.0, 1.1, 1.2, 1.5, 1.8])
    monkeypatch.setattr(
        "src.tracking.mouth.mouth_aspect_ratio",
        lambda landmarks: next(mars),
    )
    monkeypatch.setattr("src.tracking.mouth.time.time", lambda: next(times))

    assert detector.update(object(), target)[0] is False
    assert detector.update(object(), target)[0] is False
    assert detector.locked_key == target
    assert detector.update(object(), target)[0] is False

    clicked, _ = detector.update(object(), target)
    assert clicked is True
    assert detector.selected_key == target

    clicked_again, _ = detector.update(object(), target)
    assert clicked_again is False
    assert detector.selected_key is None


def test_one_frame_resolves_to_suggestion_or_key_never_both():
    rects, _ = _layout_parts()
    rect = rects[0]
    overlapping_key = Button(
        [rect.x, rect.y],
        "ㄱ",
        size=[rect.width, rect.height],
    )

    target = resolve_input_target(
        rects,
        ("감사합니다", "", ""),
        [overlapping_key],
        *rect.center,
    )

    assert target == SuggestionTarget(0, "감사합니다")
    assert not isinstance(target, str)


def test_empty_suggestion_does_not_intercept_an_overlapping_keyboard_key():
    rects, _ = _layout_parts()
    rect = rects[0]
    overlapping_key = Button(
        [rect.x, rect.y],
        "ㄱ",
        size=[rect.width, rect.height],
    )

    assert resolve_input_target(
        rects,
        ("", "", ""),
        [overlapping_key],
        *rect.center,
    ) == "ㄱ"


@pytest.mark.parametrize(
    ("prefix", "jamos", "suggestion", "expected", "expected_diff"),
    [
        ("", ["ㄱ", "ㅅ"], "감사합니다", "감사합니다", (2, "감사합니다")),
        (
            "오늘 ",
            ["ㄱ", "ㅅ"],
            "감사합니다",
            "오늘 감사합니다",
            (2, "감사합니다"),
        ),
        (
            "",
            ["ㄱ"],
            "간호사 불러주세요",
            "간호사 불러주세요",
            (1, "간호사 불러주세요"),
        ),
    ],
)
def test_suggestion_replaces_only_current_word(
    prefix,
    jamos,
    suggestion,
    expected,
    expected_diff,
):
    before = _qwerty_input(prefix, jamos)

    diff = apply_suggestion(suggestion, True, KEYBOARD_LAYOUT_QWERTY)

    assert diff == expected_diff
    assert before[:-diff[0]] + diff[1] == expected
    assert hangul.finalText == expected
    assert not hangul.finalText.endswith(" ")
    assert hangul.jamo_buffer == ['', '', '']


def test_cheonjiin_pending_and_consonant_cycle_are_cleared_after_selection():
    buttons = create_cheonjiin_buttons()

    process_key(
        "ㅇㅁ",
        True,
        False,
        buttons,
        KEYBOARD_LAYOUT_CHEONJIIN,
    )
    assert cheonjiin_composer.last_consonant is not None
    apply_suggestion("의사 불러주세요", True, KEYBOARD_LAYOUT_CHEONJIIN)
    assert not cheonjiin_composer.has_uncommitted_input()
    assert cheonjiin_composer.last_consonant is None

    hangul.finalText = ""
    hangul.jamo_buffer[:] = ['', '', '']
    cheonjiin_composer.reset()

    process_key(
        "ㆍ",
        True,
        False,
        buttons,
        KEYBOARD_LAYOUT_CHEONJIIN,
    )
    assert cheonjiin_composer.get_pending_preview() == "ㆍ"
    apply_suggestion("간호사 불러주세요", True, KEYBOARD_LAYOUT_CHEONJIIN)

    assert hangul.finalText == "간호사 불러주세요"
    assert hangul.jamo_buffer == ['', '', '']
    assert cheonjiin_composer.get_pending_preview() == ""
    assert cheonjiin_composer.vowel_tokens == []


@pytest.mark.parametrize("changed_text", ["감사합니다 ㅂ", "감사합니"])
def test_selection_stays_blank_until_new_input_or_delete_requeries(changed_text):
    calls = []

    def provider(prefix, limit):
        calls.append(prefix)
        return ["새 추천"]

    state = SuggestionStateController(suggestion_provider=provider)
    state.update("ㄱㅅ")
    state.clear_after_selection("감사합니다")

    assert state.update("감사합니다") is None
    assert state.slots == ("", "", "")

    changed = state.update(changed_text)
    assert changed is not None
    assert changed.slots == ("새 추천", "", "")
    assert len(calls) == 2


def test_selection_marks_pending_boundary_without_trailing_space():
    boundary = _select_with_pending_boundary(
        "너무 아파요",
        KEYBOARD_LAYOUT_QWERTY,
    )

    assert hangul.finalText == "너무 아파요"
    assert not hangul.finalText.endswith(" ")
    assert boundary.pending_word_boundary is True


@pytest.mark.parametrize(
    ("keyboard_layout", "key", "button_factory"),
    [
        (KEYBOARD_LAYOUT_QWERTY, "ㅂ", lambda: create_buttons(keys_kor_normal)),
        (KEYBOARD_LAYOUT_CHEONJIIN, "ㅂㅍ", create_cheonjiin_buttons),
    ],
)
def test_next_character_starts_new_word_for_qwerty_and_cheonjiin(
    keyboard_layout,
    key,
    button_factory,
):
    boundary = _select_with_pending_boundary("너무 아파요", keyboard_layout)

    result = boundary.handle_key(
        key,
        True,
        False,
        button_factory(),
        keyboard_layout,
    )

    assert _current_text(keyboard_layout) == "너무 아파요 ㅂ"
    assert result[3:] == (0, " ㅂ")
    assert boundary.pending_word_boundary is False


def test_new_word_chosung_requeries_only_the_new_character():
    calls = []

    def provider(prefix, limit):
        calls.append(prefix)
        return ["불편한 곳이 어디인가요"]

    state = SuggestionStateController(suggestion_provider=provider)
    boundary = _select_with_pending_boundary(
        "너무 아파요",
        KEYBOARD_LAYOUT_QWERTY,
    )
    state.clear_after_selection(hangul.finalText)

    boundary.handle_key(
        "ㅂ",
        True,
        False,
        create_buttons(keys_kor_normal),
        KEYBOARD_LAYOUT_QWERTY,
    )
    update = state.update(_current_text(KEYBOARD_LAYOUT_QWERTY))

    assert calls == ["ㅂ"]
    assert update.chosung_input == "ㅂ"
    assert update.slots == ("불편한 곳이 어디인가요", "", "")


def test_direct_space_inserts_exactly_one_boundary_and_clears_pending():
    boundary = _select_with_pending_boundary(
        "너무 아파요",
        KEYBOARD_LAYOUT_QWERTY,
    )

    result = boundary.handle_key(
        " ",
        True,
        False,
        create_buttons(keys_kor_normal),
        KEYBOARD_LAYOUT_QWERTY,
    )

    assert hangul.finalText == "너무 아파요 "
    assert result[3:] == (0, " ")
    assert boundary.pending_word_boundary is False


@pytest.mark.parametrize(
    ("key", "expected", "expected_diff"),
    [
        ("Del", "너무 아파", (1, "")),
        ("확인", "너무 아파요", (0, "")),
    ],
)
def test_delete_and_confirm_clear_pending_without_hidden_space(
    key,
    expected,
    expected_diff,
):
    boundary = _select_with_pending_boundary(
        "너무 아파요",
        KEYBOARD_LAYOUT_QWERTY,
    )

    result = boundary.handle_key(
        key,
        True,
        False,
        create_buttons(keys_kor_normal),
        KEYBOARD_LAYOUT_QWERTY,
    )

    assert _current_text(KEYBOARD_LAYOUT_QWERTY) == expected
    assert result[3:] == expected_diff
    assert boundary.pending_word_boundary is False


@pytest.mark.parametrize("punctuation", [",", "."])
def test_punctuation_attaches_then_next_character_starts_new_word(punctuation):
    boundary = _select_with_pending_boundary(
        "감사합니다",
        KEYBOARD_LAYOUT_QWERTY,
    )
    buttons = create_buttons(keys_kor_normal)

    first = boundary.handle_key(
        punctuation,
        True,
        False,
        buttons,
        KEYBOARD_LAYOUT_QWERTY,
    )
    assert _current_text(KEYBOARD_LAYOUT_QWERTY) == f"감사합니다{punctuation}"
    assert first[3:] == (0, punctuation)
    assert boundary.pending_word_boundary is True

    second = boundary.handle_key(
        "ㅂ",
        True,
        False,
        first[2],
        KEYBOARD_LAYOUT_QWERTY,
    )
    assert _current_text(KEYBOARD_LAYOUT_QWERTY) == (
        f"감사합니다{punctuation} ㅂ"
    )
    assert second[3:] == (0, " ㅂ")
    assert boundary.pending_word_boundary is False


def test_punctuation_keeps_slots_blank_until_new_word_character():
    calls = []

    def provider(prefix, limit):
        calls.append(prefix)
        return ["불편한 곳이 어디인가요"]

    state = SuggestionStateController(suggestion_provider=provider)
    boundary = _select_with_pending_boundary(
        "감사합니다",
        KEYBOARD_LAYOUT_QWERTY,
    )
    state.clear_after_selection(hangul.finalText)
    buttons = create_buttons(keys_kor_normal)

    boundary.handle_key(
        ".",
        True,
        False,
        buttons,
        KEYBOARD_LAYOUT_QWERTY,
    )
    assert boundary.pending_word_boundary is True
    state.clear_after_selection(_current_text(KEYBOARD_LAYOUT_QWERTY))
    assert state.slots == ("", "", "")
    assert calls == []

    boundary.handle_key(
        "ㅂ",
        True,
        False,
        buttons,
        KEYBOARD_LAYOUT_QWERTY,
    )
    update = state.update(_current_text(KEYBOARD_LAYOUT_QWERTY))
    assert calls == ["ㅂ"]
    assert update.chosung_input == "ㅂ"


def test_function_keys_do_not_create_boundary_and_language_toggle_preserves_it():
    boundary = _select_with_pending_boundary(
        "감사합니다",
        KEYBOARD_LAYOUT_QWERTY,
    )
    buttons = create_buttons(keys_kor_normal)

    is_korean, is_shift, buttons, deleted, inserted = boundary.handle_key(
        "Shift",
        True,
        False,
        buttons,
        KEYBOARD_LAYOUT_QWERTY,
    )
    assert (deleted, inserted) == (0, "")
    assert _current_text(KEYBOARD_LAYOUT_QWERTY) == "감사합니다"
    assert boundary.pending_word_boundary is True

    is_korean, is_shift, buttons, deleted, inserted = boundary.handle_key(
        "한/영",
        is_korean,
        is_shift,
        buttons,
        KEYBOARD_LAYOUT_QWERTY,
    )
    assert (deleted, inserted) == (0, "")
    assert is_korean is False
    assert boundary.pending_word_boundary is True

    result = boundary.handle_key(
        "b",
        is_korean,
        is_shift,
        buttons,
        KEYBOARD_LAYOUT_QWERTY,
    )
    assert hangul.finalText == "감사합니다 b"
    assert result[3:] == (0, " b")


def test_existing_trailing_space_is_not_duplicated_before_next_character():
    boundary = _select_with_pending_boundary(
        "감사합니다 ",
        KEYBOARD_LAYOUT_QWERTY,
    )

    result = boundary.handle_key(
        "ㅂ",
        True,
        False,
        create_buttons(keys_kor_normal),
        KEYBOARD_LAYOUT_QWERTY,
    )

    assert _current_text(KEYBOARD_LAYOUT_QWERTY) == "감사합니다 ㅂ"
    assert result[3:] == (0, "ㅂ")


@pytest.mark.parametrize("input_mode", ["dwell", "mouth"])
def test_auto_boundary_is_logged_in_one_real_character_tap(input_mode):
    logger = InputEventLogger(enabled=True, run_id=f"boundary-{input_mode}")
    boundary = _select_with_pending_boundary(
        "너무 아파요",
        KEYBOARD_LAYOUT_QWERTY,
    )
    buttons = create_buttons(keys_kor_normal)
    _, _, _, deleted, inserted = boundary.handle_key(
        "ㅂ",
        True,
        False,
        buttons,
        KEYBOARD_LAYOUT_QWERTY,
    )

    log_input_tap(
        logger,
        frame_id=8,
        input_mode=input_mode,
        keyboard_layout=KEYBOARD_LAYOUT_QWERTY,
        key_id="ㅂ",
        button_list=buttons,
        target_char="",
        hover_start_ts_ms=None,
        cursor_x=100,
        cursor_y=50,
        deleted_count=deleted,
        inserted_text=inserted,
        trigger_signal=(0.4 if input_mode == "mouth" else None),
    )

    assert len(logger._buffer) == 1
    assert logger._buffer[0]["input_mode"] == input_mode
    assert logger._buffer[0]["inserted_text"] == " ㅂ"


def test_suggestion_selection_logs_one_tap_commit_with_exact_diff():
    logger = InputEventLogger(enabled=True, run_id="suggestion-test")
    before = _qwerty_input("오늘 ", ["ㄱ", "ㅅ"])
    deleted, inserted = apply_suggestion(
        "감사합니다",
        True,
        KEYBOARD_LAYOUT_QWERTY,
    )

    log_input_tap(
        logger,
        frame_id=7,
        input_mode="dwell",
        keyboard_layout=KEYBOARD_LAYOUT_QWERTY,
        key_id="suggestion_1",
        key_label="감사합니다",
        key_center=(100, 50),
        button_list=[],
        target_char="",
        hover_start_ts_ms=None,
        cursor_x=100,
        cursor_y=50,
        deleted_count=deleted,
        inserted_text=inserted,
        trigger_signal=None,
    )

    assert len(logger._buffer) == 1
    row = logger._buffer[0]
    assert row["key_id"] == "suggestion_1"
    assert row["key_label"] == "감사합니다"
    assert row["is_backspace"] is False
    assert (row["deleted_count"], row["inserted_text"]) == (
        2,
        "감사합니다",
    )
    assert before[:-deleted] + inserted == hangul.finalText
