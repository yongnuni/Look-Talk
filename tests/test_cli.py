"""src/app/cli.py parse_args() 반환 순서와 통합 플로우 옵션 단위 테스트."""

import sys

from src.app.cli import parse_args


def test_parse_args_default_condition_label_is_empty_string(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    (
        _,
        _,
        _,
        _,
        condition_label,
        calib_points,
        performance_flow,
        autocomplete,
        test_sentence,
    ) = parse_args()

    assert condition_label == ""
    assert calib_points == 16
    assert performance_flow is False
    assert autocomplete == "on"
    assert test_sentence is None


def test_parse_args_condition_label_is_passed_through(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--condition-label", "baseline"]
    )

    _, _, _, _, condition_label, _, _, _, _ = parse_args()

    assert condition_label == "baseline"


def test_parse_args_returns_nine_values_in_order(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--keyboard-layout", "cheonjiin",
            "--user-id", "someone",
            "--condition-label", "exp-a",
            "--calib-points", "16",
            "--performance-flow",
        ],
    )

    (
        gaze_mode,
        strategy,
        keyboard_layout,
        user_id,
        condition_label,
        calib_points,
        performance_flow,
        autocomplete,
        test_sentence,
    ) = parse_args()

    assert keyboard_layout == "cheonjiin"
    assert user_id == "someone"
    assert condition_label == "exp-a"
    assert calib_points == 16
    assert performance_flow is True
    assert autocomplete == "on"
    assert test_sentence is None


def test_performance_flow_rejects_non_16_point_calibration(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--performance-flow", "--calib-points", "9"],
    )

    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("9점 통합 성능 플로우가 거부되지 않았습니다.")


def test_autocomplete_defaults_to_on(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    (*_, autocomplete, _) = parse_args()

    assert autocomplete == "on"


def test_autocomplete_off_with_condition_label_succeeds(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--autocomplete", "off", "--condition-label", "no-ac"],
    )

    result = parse_args()

    assert result[4] == "no-ac"  # condition_label
    assert result[7] == "off"  # autocomplete


def test_autocomplete_off_without_condition_label_exits(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--autocomplete", "off"]
    )

    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError(
            "--condition-label 없는 --autocomplete off가 거부되지 않았습니다."
        )


def test_test_sentence_group_and_index_resolves_to_sentence(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--test-sentence", "hit:0"]
    )

    (*_, test_sentence) = parse_args()

    assert test_sentence == "물 주세요"


def test_test_sentence_bare_index_uses_legacy_group(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--test-sentence", "1"])

    (*_, test_sentence) = parse_args()

    assert test_sentence == "감사합니다"


def test_test_sentence_unknown_group_exits(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--test-sentence", "nope:0"]
    )

    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("존재하지 않는 그룹이 거부되지 않았습니다.")


def test_test_sentence_out_of_range_index_exits(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--test-sentence", "hit:99"]
    )

    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("범위를 벗어난 인덱스가 거부되지 않았습니다.")
