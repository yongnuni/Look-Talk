"""src/app/cli.py parse_args() 반환 순서와 통합 플로우 옵션 단위 테스트."""

import sys

from src.app.cli import parse_args


def test_parse_args_default_condition_label_is_empty_string(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    _, _, _, _, condition_label, calib_points, performance_flow = parse_args()

    assert condition_label == ""
    assert calib_points == 16
    assert performance_flow is False


def test_parse_args_condition_label_is_passed_through(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--condition-label", "baseline"]
    )

    _, _, _, _, condition_label, _, _ = parse_args()

    assert condition_label == "baseline"


def test_parse_args_returns_seven_values_in_order(monkeypatch):
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
    ) = parse_args()

    assert keyboard_layout == "cheonjiin"
    assert user_id == "someone"
    assert condition_label == "exp-a"
    assert calib_points == 16
    assert performance_flow is True


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
