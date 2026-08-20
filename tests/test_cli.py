"""src/app/cli.py parse_args() 단위 테스트.

--condition-label 추가로 반환 튜플이 4개→5개가 됐다 — main.py의
__main__ 블록이 위치 기반으로 언패킹하므로, 튜플 순서/기본값이 어긋나면
condition_label 자리에 엉뚱한 값이 들어가도 조용히 통과할 수 있다.
"""

import sys

from src.app.cli import parse_args


def test_parse_args_default_condition_label_is_empty_string(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    _, _, _, _, condition_label = parse_args()

    assert condition_label == ""


def test_parse_args_condition_label_is_passed_through(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--condition-label", "baseline"]
    )

    _, _, _, _, condition_label = parse_args()

    assert condition_label == "baseline"


def test_parse_args_returns_five_values_in_order(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--keyboard-layout", "cheonjiin",
            "--user-id", "someone",
            "--condition-label", "exp-a",
        ],
    )

    gaze_mode, strategy, keyboard_layout, user_id, condition_label = parse_args()

    assert keyboard_layout == "cheonjiin"
    assert user_id == "someone"
    assert condition_label == "exp-a"
