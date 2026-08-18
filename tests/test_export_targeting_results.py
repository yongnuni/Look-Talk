"""export_targeting_results()의 fieldnames/row 스키마 일치, 출력 디렉토리
자동 생성을 검증한다.

a773ba2(2026-08-16, "입벌림 방식 개선") 이후 fieldnames가 입벌림 캘리브레이션
스키마로 잘못 교체되면서, csv.DictWriter(extrasaction 기본값 'raise')가
row 작성 시 ValueError를 던져 앱 전체가 죽는 회귀가 있었다. 이 테스트는
fieldnames와 실제 row 키가 어긋나면 즉시 실패하도록 회귀를 막는다.

모든 테스트는 tmp_path로 chdir만 하고 gaze_accuracy_results/ 디렉토리를
미리 만들어두지 않는다 — export_targeting_results()가 append_rows() 호출
전에 스스로 os.makedirs(exist_ok=True)를 하는지를 매 테스트가 자연스럽게
함께 검증한다(run_gaze_accuracy_test()가 이미 쓰는 것과 동일한 방식).

카메라 없이 돌아가야 하므로 TargetingTestRunner 전체를 구동하지 않고,
export_targeting_results()가 실제로 읽는 유일한 속성인 `.attempts`만 채운
가짜 러너 객체로 함수를 직접 호출한다.
"""

import csv
import os
from types import SimpleNamespace

import pytest

from src.testing.targeting_export import export_targeting_results
from src.common import clock
from tests.targeting_test_runner import TargetAttempt


@pytest.fixture(autouse=True)
def _clock_init():
    clock._reset()
    clock.init()
    yield
    clock._reset()


def _make_attempt(trial, success=True, reason="dwell"):
    return TargetAttempt(
        trial=trial,
        target_x=100 + trial,
        target_y=200 + trial,
        selected_x=110 + trial,
        selected_y=210 + trial,
        success=success,
        reaction_time_sec=1.234,
        reason=reason,
    )


def _read_csv_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def test_export_targeting_results_creates_missing_directory(tmp_path, monkeypatch):
    """gaze_accuracy_results/가 아예 없는 상태에서도 함수가 스스로 만들어야 한다."""
    monkeypatch.chdir(tmp_path)

    out_dir = tmp_path / "gaze_accuracy_results"
    assert not out_dir.exists()

    runner = SimpleNamespace(attempts=[_make_attempt(1)])

    export_targeting_results(
        run_id="run-test-0",
        keyboard_layout="qwerty",
        targeting_runner=runner,
        aborted=False,
    )

    assert out_dir.is_dir()
    assert (out_dir / "targeting_results_v1.0.csv").is_file()


def test_export_targeting_results_writes_without_crashing(tmp_path, monkeypatch):
    """fieldnames와 row 키가 어긋나면 append_rows가 ValueError를 던진다 —
    예외 없이 끝까지 실행되는 것 자체가 회귀 검증이다."""
    monkeypatch.chdir(tmp_path)

    runner = SimpleNamespace(
        attempts=[
            _make_attempt(1, success=True, reason="dwell"),
            _make_attempt(2, success=False, reason="timeout"),
            _make_attempt(3, success=True, reason="selection_hit"),
        ]
    )

    export_targeting_results(
        run_id="run-test-1",
        keyboard_layout="qwerty",
        targeting_runner=runner,
        aborted=False,
    )

    csv_path = tmp_path / "gaze_accuracy_results" / "targeting_results_v1.0.csv"
    assert csv_path.is_file()


def test_export_targeting_results_fieldnames_match_row_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    runner = SimpleNamespace(attempts=[_make_attempt(1)])

    export_targeting_results(
        run_id="run-test-2",
        keyboard_layout="cheonjiin",
        targeting_runner=runner,
        aborted=False,
    )

    csv_path = os.path.join(
        str(tmp_path), "gaze_accuracy_results", "targeting_results_v1.0.csv"
    )
    fieldnames, rows = _read_csv_rows(csv_path)

    expected_keys = {
        "run_id",
        "keyboard_layout",
        "ts_ms",
        "target_index",
        "success",
        "reaction_time_sec",
        "input_mode",
        "timeout",
        "target_x",
        "target_y",
        "cursor_x",
        "cursor_y",
        "aborted",
    }

    assert set(fieldnames) == expected_keys
    assert len(rows) == 1
    assert set(rows[0].keys()) == expected_keys


def test_export_targeting_results_records_aborted_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    runner = SimpleNamespace(attempts=[_make_attempt(1)])

    export_targeting_results(
        run_id="run-test-3",
        keyboard_layout="qwerty",
        targeting_runner=runner,
        aborted=True,
    )

    csv_path = os.path.join(
        str(tmp_path), "gaze_accuracy_results", "targeting_results_v1.0.csv"
    )
    _, rows = _read_csv_rows(csv_path)

    assert rows[0]["aborted"] == "True"


def test_export_targeting_results_skips_empty_attempts(tmp_path, monkeypatch):
    """attempts가 비어 있으면(예: 타겟팅 진입 직후 ESC) 아무 파일도 만들지 않는다."""
    monkeypatch.chdir(tmp_path)

    runner = SimpleNamespace(attempts=[])

    export_targeting_results(
        run_id="run-test-4",
        keyboard_layout="qwerty",
        targeting_runner=runner,
        aborted=True,
    )

    out_dir = tmp_path / "gaze_accuracy_results"
    assert not out_dir.exists()
    assert not (out_dir / "targeting_results_v1.0.csv").exists()
