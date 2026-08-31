import csv
import os

import pytest

from src.common import clock
from src.metrics.blink_event_logger import BlinkClosureEarTracker, BlinkEventLogger
from src.tracking.blink import BlinkEvent, BlinkKind


def test_ear_tracker_returns_current_ear_when_never_closed():
    tracker = BlinkClosureEarTracker()

    assert tracker.observe(False, 0.30) == 0.30
    assert tracker.observe(False, 0.28) == 0.28


def test_ear_tracker_returns_minimum_reached_during_closure_on_reopen():
    tracker = BlinkClosureEarTracker()

    assert tracker.observe(True, 0.20) == 0.20
    assert tracker.observe(True, 0.12) == 0.12
    assert tracker.observe(True, 0.15) == 0.12  # 최솟값 유지, 0.15로 덮이지 않음

    # 재개안(is_closed=False) 프레임 - 이 프레임의 ear(0.25)가 아니라 감음 구간
    # 최솟값(0.12)이 반환돼야 한다(BlinkDetector는 이 프레임에 이벤트를 낸다).
    assert tracker.observe(False, 0.25) == 0.12


def test_ear_tracker_resets_for_next_closure():
    tracker = BlinkClosureEarTracker()

    tracker.observe(True, 0.20)
    tracker.observe(True, 0.12)
    tracker.observe(False, 0.25)  # 첫 구간 소비 + 리셋

    # 두 번째 구간은 이전 최솟값(0.12)에 오염되지 않고 새로 시작해야 한다.
    assert tracker.observe(True, 0.18) == 0.18


def test_ear_tracker_keeps_tracking_through_long_closure():
    tracker = BlinkClosureEarTracker()

    tracker.observe(True, 0.20)
    # LONG_CLOSURE가 발화해도 눈은 계속 감긴 상태이므로 is_closed=True가 유지된다.
    assert tracker.observe(True, 0.10) == 0.10
    assert tracker.observe(True, 0.16) == 0.10  # 최솟값 계속 유지


@pytest.fixture(autouse=True)
def _clock_init():
    clock._reset()
    clock.init()
    yield
    clock._reset()


def _make_logger(tmp_path, **kwargs):
    path = os.path.join(str(tmp_path), "blink_events_test.csv")
    return BlinkEventLogger(path=path, run_id="run_test", **kwargs), path


def _event(kind=BlinkKind.NATURAL, duration=0.187):
    return BlinkEvent(kind=kind, duration=duration, timestamp=1.0)


def test_none_event_is_ignored(tmp_path):
    logger, _ = _make_logger(tmp_path, flush_every=1)

    logger.log_event(None, ear_at_close=0.15)

    assert logger._buffer == []


def test_log_event_converts_duration_to_ms_and_kind_to_name(tmp_path):
    logger, _ = _make_logger(tmp_path, flush_every=100)

    logger.log_event(_event(BlinkKind.NATURAL, duration=0.187), ear_at_close=0.1366)

    row = logger._buffer[0]
    assert row["kind"] == "NATURAL"
    assert row["duration_ms"] == pytest.approx(187.0)
    assert row["ear_at_close"] == 0.1366
    assert row["run_id"] == "run_test"


def test_flush_writes_buffer_and_clears_it(tmp_path):
    logger, path = _make_logger(tmp_path, flush_every=100)

    logger.log_event(_event(BlinkKind.NATURAL), ear_at_close=0.14)
    logger.log_event(_event(BlinkKind.INTENTIONAL, duration=0.5), ear_at_close=0.13)
    assert len(logger._buffer) == 2

    logger.flush()

    assert logger._buffer == []
    assert os.path.isfile(path)

    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert set(BlinkEventLogger.FIELDNAMES) <= set(rows[0].keys())


def test_auto_flush_at_threshold(tmp_path):
    logger, path = _make_logger(tmp_path, flush_every=3)

    for _ in range(3):
        logger.log_event(_event(), ear_at_close=0.14)

    assert logger._buffer == []
    assert os.path.isfile(path)

    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3


def test_close_flushes_pending_buffer(tmp_path):
    logger, path = _make_logger(tmp_path, flush_every=100)

    logger.log_event(_event(), ear_at_close=0.14)
    logger.close()

    assert os.path.isfile(path)
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1


def test_write_failure_warns_once_and_counts_drops(tmp_path, monkeypatch, capsys):
    logger, _ = _make_logger(tmp_path, flush_every=1)

    def _raise(*args, **kwargs):
        raise OSError("디스크 오류 시뮬레이션")

    monkeypatch.setattr("src.metrics.csv_export.append_rows", _raise)

    logger.log_event(_event(), ear_at_close=0.14)  # flush_every=1 -> 즉시 flush 실패
    logger.log_event(_event(), ear_at_close=0.14)  # 이후는 조용히 드롭 카운트만 증가
    logger.log_event(_event(), ear_at_close=0.14)

    captured = capsys.readouterr()
    warning_lines = [
        line for line in captured.out.splitlines()
        if "[blink_event_logger]" in line and "CSV 쓰기 실패" in line
    ]
    assert len(warning_lines) == 1

    assert logger._write_failed is True
    assert logger._dropped_event_count == 3

    capsys.readouterr()
    logger.close()
    close_out = capsys.readouterr().out
    assert "유실된 이벤트 3건" in close_out


def test_disabled_logger_is_a_no_op(tmp_path):
    logger, path = _make_logger(tmp_path, enabled=False)

    logger.log_event(_event(), ear_at_close=0.14)
    logger.close()

    assert logger._buffer == []
    assert not os.path.isfile(path)
