import csv
import os

import pytest

from src.common import clock
from src.metrics.input_event_logger import InputEventLogger


@pytest.fixture(autouse=True)
def _clock_init():
    clock._reset()
    clock.init()
    yield
    clock._reset()


def _make_logger(tmp_path, **kwargs):
    path = os.path.join(str(tmp_path), "input_events_test.csv")
    return InputEventLogger(path=path, run_id="run_test", **kwargs), path


def _log_one(logger, **overrides):
    kwargs = dict(
        frame_id=1,
        input_mode="dwell",
        keyboard_layout="qwerty",
        key_id="ㄱ",
        key_label="ㄱ",
        is_backspace=False,
        deleted_count=0,
        inserted_text="ㄱ",
        hover_to_commit_ms=1200.0,
        cursor_x=10,
        cursor_y=20,
        key_center_x=15,
        key_center_y=25,
        target_char="ㄱ",
        trigger_signal=None,
    )
    kwargs.update(overrides)
    logger.log_tap_commit(**kwargs)


def test_event_seq_is_monotonic(tmp_path):
    logger, _ = _make_logger(tmp_path, flush_every=100)

    for i in range(5):
        _log_one(logger, frame_id=i)

    seqs = [row["event_seq"] for row in logger._buffer]
    assert seqs == [0, 1, 2, 3, 4]


def test_flush_writes_buffer_and_clears_it(tmp_path):
    logger, path = _make_logger(tmp_path, flush_every=100)

    _log_one(logger)
    _log_one(logger)
    assert len(logger._buffer) == 2

    logger.flush()

    assert logger._buffer == []
    assert os.path.isfile(path)

    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert set(InputEventLogger.FIELDNAMES) <= set(rows[0].keys())


def test_auto_flush_at_threshold(tmp_path):
    logger, path = _make_logger(tmp_path, flush_every=3)

    for i in range(3):
        _log_one(logger, frame_id=i)

    # 3번째 기록에서 자동 flush가 이미 발생했어야 한다.
    assert logger._buffer == []
    assert os.path.isfile(path)

    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3


def test_close_flushes_pending_buffer(tmp_path):
    logger, path = _make_logger(tmp_path, flush_every=100)

    _log_one(logger)
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

    _log_one(logger, frame_id=1)  # flush_every=1이므로 즉시 flush 시도 -> 실패
    _log_one(logger, frame_id=2)  # 이후 호출은 write_failed라 조용히 드롭 카운트만 증가
    _log_one(logger, frame_id=3)

    captured = capsys.readouterr()
    warning_lines = [
        line for line in captured.out.splitlines()
        if "[input_event_logger]" in line and "CSV 쓰기 실패" in line
    ]
    assert len(warning_lines) == 1  # 경고는 최초 1회만

    assert logger._write_failed is True
    assert logger._dropped_event_count == 3  # 1개(flush 실패분) + 2개(드롭)

    capsys.readouterr()
    logger.close()
    close_out = capsys.readouterr().out
    assert "유실된 이벤트 3건" in close_out


def test_disabled_logger_is_a_no_op(tmp_path):
    logger, path = _make_logger(tmp_path, enabled=False)

    _log_one(logger)
    logger.close()

    assert logger._buffer == []
    assert not os.path.isfile(path)
