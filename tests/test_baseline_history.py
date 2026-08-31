"""append_mouth_baseline_history()가 출력 디렉토리(calibration_results/)를
스스로 보장하는지 검증한다.

이전에는 directly save_baseline()이 같은 디렉토리를 만들어주는 부수효과에
기대는 간접 의존 상태였다(main.py에서 항상 save_baseline() 직후에만
호출됐기 때문에 우연히 안전했음). 이 테스트는 그 간접 의존을 없애고
append_mouth_baseline_history() 스스로 os.makedirs를 호출하는지 확인한다.
"""

import csv
import json

import pytest

from src.calibrations.baseline_manager import save_baseline
from src.common import clock
from src.metrics.baseline_history import (
    append_blink_baseline_history,
    append_mouth_baseline_history,
)


@pytest.fixture(autouse=True)
def _clock_init():
    clock._reset()
    clock.init()
    yield
    clock._reset()


def _read_csv_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_append_mouth_baseline_history_creates_missing_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # baseline.json은 append_mouth_baseline_history가 만드는 출력 디렉토리
    # (calibration_results/)와 다른 경로에 둔다 — save_baseline()이 자기 몫의
    # 디렉토리를 만드는 것과 이번에 검증하려는 동작을 섞지 않기 위함.
    baseline_path = tmp_path / "baseline_src" / "baseline.json"
    saved_path = save_baseline(
        mouth_result={"open_threshold": 0.3, "close_threshold": 0.2},
        path=str(baseline_path),
    )

    out_dir = tmp_path / "calibration_results"
    assert not out_dir.exists()

    append_mouth_baseline_history(run_id="run-test", saved_path=saved_path)

    csv_path = out_dir / "mouth_baseline_history_v2.0.csv"
    assert csv_path.is_file()

    rows = _read_csv_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-test"
    assert rows[0]["open_threshold"] == "0.3"
    assert rows[0]["close_threshold"] == "0.2"


def test_combined_flow_preserves_blink_and_mouth_in_existing_baseline(tmp_path):
    baseline_path = tmp_path / "combined" / "baseline.json"
    save_baseline(
        mouth_result={"open_threshold": 0.3, "close_threshold": 0.2},
        blink_result={"close_threshold": 0.17, "open_threshold": 0.21},
        path=str(baseline_path),
    )

    with open(baseline_path, encoding="utf-8") as handle:
        baseline = json.load(handle)

    assert baseline["mouth"]["open_threshold"] == 0.3
    assert baseline["blink"]["close_threshold"] == 0.17


def test_append_blink_baseline_history_writes_separate_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    baseline_path = tmp_path / "baseline_src" / "baseline.json"
    saved_path = save_baseline(
        mouth_result={"open_threshold": 0.3, "close_threshold": 0.2},
        blink_result={
            "open_ear_median": 0.31,
            "closed_ear_median": 0.11,
            "close_threshold": 0.17,
            "open_threshold": 0.21,
            "calibration_failed_fallback": False,
            "total_trials": 5,
            "closed_sample_count": 5,
        },
        path=str(baseline_path),
    )

    append_blink_baseline_history(
        run_id="run-test", saved_path=saved_path, user_id="tester"
    )

    csv_path = tmp_path / "calibration_results" / "blink_baseline_history_v1.0.csv"
    assert csv_path.is_file()

    rows = _read_csv_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-test"
    assert rows[0]["user_id"] == "tester"
    assert rows[0]["open_ear_median"] == "0.31"
    assert rows[0]["close_threshold"] == "0.17"
    assert rows[0]["fallback"] == "False"

    # mouth 이력 파일 스키마는 그대로여야 한다(별도 파일로 분리).
    mouth_csv_path = (
        tmp_path / "calibration_results" / "mouth_baseline_history_v2.0.csv"
    )
    assert not mouth_csv_path.is_file()


def test_append_blink_baseline_history_is_noop_without_blink_result(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    baseline_path = tmp_path / "baseline_src" / "baseline.json"
    saved_path = save_baseline(
        mouth_result={"open_threshold": 0.3, "close_threshold": 0.2},
        path=str(baseline_path),
    )

    append_blink_baseline_history(
        run_id="run-test", saved_path=saved_path, user_id="tester"
    )

    csv_path = tmp_path / "calibration_results" / "blink_baseline_history_v1.0.csv"
    assert not csv_path.is_file()
