"""src/metrics/derive_input.py 검증.

합성 delta 시퀀스로 파생 지표(replay/tap_count/taps_per_char/error_rate/
hover_to_commit_ms)를 확인한다. QWERTY 종성 소급 재해석 케이스를 반드시
포함해 char_count가 "탭 수"가 아니라 "replay 최종 문자열 길이" 기준으로
나오는지 확인한다(3단계 작업 지시 핵심 요구사항).
"""

import csv

import pytest

from src.metrics import derive_input
from src.metrics.input_event_logger import InputEventLogger


def _write_events_csv(path, rows):
    """FIELDNAMES 순서에 맞춰 input_events CSV를 합성해서 쓴다.

    rows의 각 dict는 필요한 컬럼만 채우면 되고, 나머지는 빈 문자열로
    채워진다(실제 InputEventLogger가 쓰는 형식과 동일 - utf-8-sig, 빈 문자열).
    """
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=InputEventLogger.FIELDNAMES)
        writer.writeheader()
        for row in rows:
            full_row = {name: row.get(name, "") for name in InputEventLogger.FIELDNAMES}
            writer.writerow(full_row)


def _tap(run_id, event_seq, deleted_count, inserted_text, input_mode="dwell",
         keyboard_layout="qwerty", is_backspace=False, hover_to_commit_ms=""):
    return {
        "run_id": run_id,
        "ts_ms": event_seq * 1000,
        "frame_id": event_seq,
        "event_seq": event_seq,
        "event_type": InputEventLogger.EVENT_TAP_COMMIT,
        "input_mode": input_mode,
        "keyboard_layout": keyboard_layout,
        "key_id": "x",
        "key_label": "x",
        "is_backspace": is_backspace,
        "deleted_count": deleted_count,
        "inserted_text": inserted_text,
        "hover_to_commit_ms": hover_to_commit_ms,
    }


# ── replay_deltas 자체 ──────────────────────────────────────────


def test_replay_deltas_pure_append():
    assert derive_input.replay_deltas([(0, "ㄱ"), (0, "가")]) == "ㄱ가"


def test_replay_deltas_backspace_unwind():
    diffs = [(0, "ㄱ"), (1, "가"), (1, "ㄱ"), (1, "")]
    assert derive_input.replay_deltas(diffs) == ""


def test_replay_deltas_qwerty_retroactive_jongsung_reinterpretation():
    """"ㄱㅏㄴㅏ" 입력 -> "가나" (tests/test_keyboard.py와 동일 케이스).

    탭 수는 4인데 최종 문자열 길이는 2 - char_count가 탭 수를 그대로
    쓰면 안 된다는 걸 보여주는 핵심 케이스.
    """
    diffs = [(0, "ㄱ"), (1, "가"), (1, "간"), (1, "가나")]
    result = derive_input.replay_deltas(diffs)
    assert result == "가나"
    assert len(result) == 2
    assert len(diffs) == 4


# ── load_input_events: 결측 vs 빈 문자열 구분 ──────────────────────


def test_load_input_events_distinguishes_empty_string_from_missing(tmp_path):
    path = tmp_path / "input_events_v1.0.csv"
    rows = [
        # 순수 백스페이스 탭: inserted_text가 실제로 빈 문자열
        _tap("r1", 0, 1, "", is_backspace=True, hover_to_commit_ms=800.0),
        # hover_to_commit_ms가 진짜 결측(dwell_start 없이 들어온 경우 없음이지만
        # 방어적으로 검증)
        _tap("r1", 1, 0, "ㄱ", hover_to_commit_ms=""),
    ]
    _write_events_csv(path, rows)

    df = derive_input.load_input_events(str(path))

    # 빈 문자열 inserted_text는 NaN이 아니라 "" 그대로 남아야 replay가
    # 이 탭을 "아무것도 안 붙임"으로 정확히 처리한다.
    assert df.loc[0, "inserted_text"] == ""
    assert df["inserted_text"].isna().sum() == 0

    # hover_to_commit_ms는 숫자 컬럼이라 빈 문자열이 NaN으로 변환돼야 한다.
    assert df.loc[0, "hover_to_commit_ms"] == 800.0
    import pandas as pd
    assert pd.isna(df.loc[1, "hover_to_commit_ms"])

    # is_backspace는 문자열 "True"/"False"가 아니라 실제 bool이어야 한다.
    assert df.loc[0, "is_backspace"] == True  # noqa: E712
    assert df.loc[1, "is_backspace"] == False  # noqa: E712
    assert isinstance(df.loc[0, "is_backspace"].item(), bool)


# ── derive_run_summary / derive_group_metrics ──────────────────────


@pytest.fixture
def events_df(tmp_path):
    path = tmp_path / "input_events_v1.0.csv"
    rows = [
        _tap("r1", 0, 0, "ㄱ", input_mode="dwell", hover_to_commit_ms=1200.0),
        _tap("r1", 1, 1, "가", input_mode="dwell", hover_to_commit_ms=900.0),
        _tap("r1", 2, 1, "간", input_mode="mouth", hover_to_commit_ms=700.0),
        _tap("r1", 3, 1, "가나", input_mode="mouth", hover_to_commit_ms=500.0),
        _tap("r1", 4, 1, "", input_mode="mouth", is_backspace=True, hover_to_commit_ms=600.0),
        # 다른 run - 섞이면 안 된다.
        _tap("r2", 0, 0, "ㄴ", input_mode="dwell"),
    ]
    _write_events_csv(path, rows)
    return derive_input.load_input_events(str(path))


def test_derive_run_summary_char_count_from_replay_not_tap_count(events_df):
    summary = derive_input.derive_run_summary(events_df, "r1")

    # 탭 5회(백스페이스 포함) 후 "가나"에서 마지막 백스페이스로 "가"만 남음.
    assert summary["tap_count"] == 5
    assert summary["backspace_count"] == 1
    assert summary["final_text"] == "가"
    assert summary["char_count"] == 1
    assert summary["tap_count"] != summary["char_count"]


def test_derive_run_summary_isolates_other_run_ids(events_df):
    summary = derive_input.derive_run_summary(events_df, "r2")
    assert summary["tap_count"] == 1
    assert summary["final_text"] == "ㄴ"


def test_derive_run_summary_taps_per_char_and_keyboard_layout(events_df):
    summary = derive_input.derive_run_summary(events_df, "r1")

    # tap_count=5, backspace_count=1, char_count=1 (r1 fixture 기준)
    assert summary["keyboard_layout"] == "qwerty"
    assert summary["taps_per_char_incl_backspace"] == pytest.approx(5 / 1)
    assert summary["taps_per_char_excl_backspace"] == pytest.approx((5 - 1) / 1)


def test_derive_run_summary_taps_per_char_none_when_char_count_zero():
    """char_count가 0(replay 결과가 빈 문자열)이면 0으로 나누지 않고 None."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "input_events_v1.0.csv")
        rows = [
            _tap("r4", 0, 0, "ㄱ", input_mode="dwell"),
            _tap("r4", 1, 1, "", input_mode="dwell", is_backspace=True),
        ]
        _write_events_csv(path, rows)
        df = derive_input.load_input_events(path)

    summary = derive_input.derive_run_summary(df, "r4")
    assert summary["char_count"] == 0
    assert summary["taps_per_char_incl_backspace"] is None
    assert summary["taps_per_char_excl_backspace"] is None


def test_derive_group_metrics_splits_by_mode_with_tap_share_pct(events_df):
    """taps_per_char/final_text는 derive_group_metrics가 아니라
    derive_run_summary가 낸다(확정 문자는 특정 모드에 귀속 불가) -
    모드별 행에는 대신 tap_share_pct(전체 탭 중 비중)가 붙는다.
    """
    rows = derive_input.derive_group_metrics(events_df, "r1")
    by_mode = {r["input_mode"]: r for r in rows}

    assert set(by_mode.keys()) == {"dwell", "mouth"}

    # dwell: 2탭 (ㄱ, 가), mouth: 3탭 (간, 가나, 백스페이스 1회) - 총 5탭
    assert by_mode["dwell"]["tap_count"] == 2
    assert by_mode["dwell"]["backspace_count"] == 0
    assert by_mode["mouth"]["tap_count"] == 3
    assert by_mode["mouth"]["backspace_count"] == 1

    assert by_mode["dwell"]["tap_share_pct"] == pytest.approx(2 / 5 * 100)
    assert by_mode["mouth"]["tap_share_pct"] == pytest.approx(3 / 5 * 100)
    assert "final_text" not in by_mode["dwell"]
    assert "taps_per_char_incl_backspace" not in by_mode["dwell"]


def test_derive_group_metrics_error_rate():
    # error_rate = backspace_count / tap_count (탭 기준, ACC-02)
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "input_events_v1.0.csv")
        rows = [
            _tap("r3", 0, 0, "ㄱ", input_mode="dwell"),
            _tap("r3", 1, 0, "ㄴ", input_mode="dwell"),
            _tap("r3", 2, 1, "", input_mode="dwell", is_backspace=True),
            _tap("r3", 3, 1, "", input_mode="dwell", is_backspace=True),
        ]
        _write_events_csv(path, rows)
        df = derive_input.load_input_events(path)

    grouped = derive_input.derive_group_metrics(df, "r3")
    dwell = grouped[0]
    assert dwell["tap_count"] == 4
    assert dwell["backspace_count"] == 2
    assert dwell["error_rate"] == pytest.approx(2 / 4)


def test_derive_group_metrics_hover_to_commit_ms_mean_and_median(events_df):
    rows = derive_input.derive_group_metrics(events_df, "r1")
    dwell = next(r for r in rows if r["input_mode"] == "dwell")

    # dwell 탭 hover_to_commit_ms: 1200.0, 900.0
    assert dwell["hover_to_commit_ms_mean"] == pytest.approx((1200.0 + 900.0) / 2)
    assert dwell["hover_to_commit_ms_median"] == pytest.approx((1200.0 + 900.0) / 2)


# ── find_sentence_completion ────────────────────────────────────────


def test_find_sentence_completion_finds_first_match_and_ignores_later_typing():
    """완주 이후 같은 run에서 계속 타이핑해도 첫 일치 시점만 잡아야 한다.

    sessions가 더 이상 완주 시점 탭 수를 저장하지 않으므로(3단계 관문 통과
    후 TestRunner 집계 제거), 대조 B는 이 함수로 완주 시점을 사후에 찾는다.
    """
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "input_events_v1.0.csv")
        rows = [
            _tap("r5", 0, 0, "가", input_mode="dwell"),
            _tap("r5", 1, 0, "나", input_mode="dwell"),  # "가나" - 목표 문장 완성
            _tap("r5", 2, 0, "다", input_mode="mouth"),  # 완주 후 이어 친 글자
        ]
        _write_events_csv(path, rows)
        df = derive_input.load_input_events(path)

    match = derive_input.find_sentence_completion(df, "r5", ["가나"])

    assert match == {"tap_count": 2, "text": "가나"}


def test_find_sentence_completion_returns_none_when_no_match():
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "input_events_v1.0.csv")
        rows = [_tap("r6", 0, 0, "가", input_mode="dwell")]
        _write_events_csv(path, rows)
        df = derive_input.load_input_events(path)

    assert derive_input.find_sentence_completion(df, "r6", ["가나"]) is None
