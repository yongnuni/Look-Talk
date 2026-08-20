"""MetricsCollector.export_csv()가 sessions_path/accuracy_path의 부모 디렉토리를
스스로 보장하는지 검증한다.

main.py의 종료 시점 안전망 호출(export_csv, "9점 테스트 아예 안 한 경우")은
디렉토리 생성을 자체적으로 보장하지 않았다 — 다른 로거가 먼저 실행돼
gaze_accuracy_results/를 만들어줬을 것이라는 간접 가정에 기대고 있었고,
그 가정이 깨지면(극단적으로 짧은 실행 등) try/except에 잡혀 세션 행이
조용히 유실될 수 있었다. 이 테스트는 export_csv() 스스로가 os.makedirs를
호출해 그 가정을 없앴는지 확인한다.

os.makedirs("", exist_ok=True)는 FileNotFoundError를 던지므로(윈도우/리눅스
공통), sessions_path/accuracy_path가 디렉토리 없이 파일명만으로 넘어오는
경우(os.path.dirname이 빈 문자열)를 건너뛰는지도 함께 검증한다.
"""

import csv

import pytest

from src.metrics.collector import MetricsCollector


def _make_collector():
    return MetricsCollector(user_id="tester", run_id="run-test")


def test_export_csv_creates_missing_sessions_directory(tmp_path):
    collector = _make_collector()

    sessions_path = tmp_path / "nested" / "dir" / "sessions_test.csv"
    accuracy_path = tmp_path / "nested" / "dir" / "gaze_accuracy_test.csv"
    assert not sessions_path.parent.exists()

    collector.export_csv(
        sessions_path=str(sessions_path),
        accuracy_path=str(accuracy_path),
        export_session=True,
        export_accuracy=False,
        export_reason=MetricsCollector.EXPORT_REASON_APP_EXIT,
    )

    assert sessions_path.is_file()


def test_export_csv_creates_missing_accuracy_directory(tmp_path):
    collector = _make_collector()

    # target_rows에 최소 1행을 채운다 (add_sample 없이도 end_target()이
    # valid_count==0 분기로 accuracy_fields와 정확히 일치하는 행을 만든다).
    collector.start_target(0, 100, 200)
    collector.end_target()

    sessions_path = tmp_path / "another" / "nested" / "sessions_test.csv"
    accuracy_path = tmp_path / "another" / "nested" / "gaze_accuracy_test.csv"
    assert not accuracy_path.parent.exists()

    collector.export_csv(
        sessions_path=str(sessions_path),
        accuracy_path=str(accuracy_path),
        export_session=False,
        export_accuracy=True,
    )

    assert accuracy_path.is_file()


def test_export_csv_skips_makedirs_for_bare_filename(tmp_path, monkeypatch):
    """디렉토리 없이 파일명만 넘어오면(os.path.dirname == "") makedirs를
    건너뛴다. 건너뛰지 않으면 os.makedirs("", exist_ok=True)가
    FileNotFoundError를 던진다."""
    monkeypatch.chdir(tmp_path)
    collector = _make_collector()

    collector.export_csv(
        sessions_path="bare_sessions_test.csv",
        accuracy_path="bare_accuracy_test.csv",
        export_session=True,
        export_accuracy=False,
        export_reason=MetricsCollector.EXPORT_REASON_APP_EXIT,
    )

    assert (tmp_path / "bare_sessions_test.csv").is_file()


def test_export_csv_writes_git_commit_and_condition_label(tmp_path):
    """git_commit/condition_label(v1.9 신규 컬럼)이 sessions 행에 그대로
    기록되는지 확인한다 — 둘 다 main.py가 계산해서 넘겨주는 값을 저장만
    할 뿐이므로, 생성자로 받은 값이 CSV에 그대로 나오면 배선이 맞다."""
    collector = MetricsCollector(
        user_id="tester",
        run_id="run-test",
        git_commit="abc1234-dirty",
        condition_label="new-smoothing",
    )

    sessions_path = tmp_path / "sessions_test.csv"
    collector.export_csv(
        sessions_path=str(sessions_path),
        accuracy_path=str(tmp_path / "gaze_accuracy_test.csv"),
        export_session=True,
        export_accuracy=False,
        export_reason=MetricsCollector.EXPORT_REASON_APP_EXIT,
    )

    with open(sessions_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["git_commit"] == "abc1234-dirty"
    assert rows[0]["condition_label"] == "new-smoothing"


def test_export_csv_defaults_git_commit_and_condition_label_to_empty(tmp_path):
    """생성자에 git_commit/condition_label을 넘기지 않으면(예: 이 파일의
    기존 _make_collector 헬퍼) None이 저장되지, 예외 없이 조용히 빠지지
    않는지 확인한다."""
    collector = _make_collector()

    sessions_path = tmp_path / "sessions_test.csv"
    collector.export_csv(
        sessions_path=str(sessions_path),
        accuracy_path=str(tmp_path / "gaze_accuracy_test.csv"),
        export_session=True,
        export_accuracy=False,
        export_reason=MetricsCollector.EXPORT_REASON_APP_EXIT,
    )

    with open(sessions_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["git_commit"] == ""
    assert rows[0]["condition_label"] == ""
