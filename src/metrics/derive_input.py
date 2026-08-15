"""input_events_v1.0.csv(2단계) 원시 로그에서 입력 지표를 사후 파생하는 순수 함수 모듈.

이 모듈은 CSV를 읽어 계산만 한다 - main.py 루프나 로거에 배선되지 않는다.
"char_commit" 이벤트가 존재하지 않는 이유(한글/천지인 조합의 소급 재해석)는
src/metrics/input_event_logger.py 모듈 docstring 및
tests/test_keyboard.py 참고. 그래서 "확정 문자열"은 tap마다 남긴 꼬리 diff
(deleted_count, inserted_text)를 재생(replay)해서만 얻을 수 있다.

replay_deltas()는 tests/test_keyboard.py의 _replay(), scripts/verify_stage1.py의
_replay_tail_diffs()와 같던 로직을 이 모듈로 통합한 것이다 - 두 파일 모두
여기서 import해서 쓴다(3단계 작업 지시).
"""

import pandas as pd

from src.metrics.input_event_logger import InputEventLogger

# derive_run_summary/derive_group_metrics가 기대하는 컬럼. 결측 처리 방식이
# 컬럼마다 다르므로(문자열 컬럼의 빈 문자열은 결측이 아님, 숫자 컬럼의
# 빈 문자열은 결측) 두 그룹으로 나눠 관리한다.
_NUMERIC_COLUMNS = [
    "ts_ms", "frame_id", "event_seq", "deleted_count",
    "hover_to_commit_ms", "cursor_x", "cursor_y",
    "key_center_x", "key_center_y",
]
_STRING_COLUMNS = [
    "run_id", "event_type", "input_mode", "keyboard_layout",
    "key_id", "key_label", "inserted_text", "target_char", "trigger_signal",
]


def _apply_delta(composite, deleted_count, inserted_text):
    """composite 문자열에 (deleted_count, inserted_text) 델타 1건을 적용한다.

    deleted_count가 NaN/None이면 0으로, inserted_text가 NaN/None이면 빈
    문자열로 본다(pandas 결측치를 안전하게 다루기 위함 - 실제 기록된 빈
    문자열 ""과는 load_input_events()에서 이미 구분해뒀으므로 여기
    도달하는 NaN은 진짜 결측이다).
    """
    deleted_count = int(deleted_count) if pd.notna(deleted_count) else 0
    inserted_text = str(inserted_text) if pd.notna(inserted_text) else ""
    if deleted_count > 0:
        composite = (
            composite[:-deleted_count]
            if deleted_count <= len(composite)
            else ""
        )
    return composite + inserted_text


def replay_deltas(deltas):
    """(deleted_count, inserted_text) 시퀀스를 재생해 최종 문자열을 복원한다.

    src/keyboard.py의 _diff_tail이 만드는 형식과 동일하다: after ==
    before[:len(before) - deleted_count] + inserted_text.
    deltas는 (deleted_count, inserted_text) 튜플의 iterable.
    """
    composite = ""
    for deleted_count, inserted_text in deltas:
        composite = _apply_delta(composite, deleted_count, inserted_text)
    return composite


def find_sentence_completion(df, run_id, candidates):
    """run_id의 탭을 순서대로 replay하면서, 매 탭 이후 상태가 candidates
    (문자열 집합/iterable) 중 하나와 처음 일치하는 시점을 찾는다.

    sessions는 "완주 시점까지 탭이 몇 개였는지"를 더 이상 저장하지 않는다
    (TestRunner의 keystrokes 집계 제거 - 3단계 관문 통과 후). 그래서 문장
    완주 여부/시점을 input_events만으로 사후에 알아내려면 이렇게 순서대로
    replay하며 목표 문장 후보와의 일치 여부를 매 탭마다 확인하는 수밖에
    없다 - 완주 이후에 이어 친 글자가 있어도(같은 run에서 계속 타이핑)
    첫 일치 시점만 취하므로 섞이지 않는다.

    반환: {"tap_count": 일치까지의 탭 수(1부터 시작), "text": 일치한 문자열}
    또는 끝까지 일치하는 시점이 없으면 None.
    """
    run_df = sorted_run_events(df, run_id)
    candidates = set(candidates)
    composite = ""

    for i, (deleted_count, inserted_text) in enumerate(
        zip(run_df["deleted_count"], run_df["inserted_text"]), start=1
    ):
        composite = _apply_delta(composite, deleted_count, inserted_text)
        if composite in candidates:
            return {"tap_count": i, "text": composite}

    return None


def load_input_events(path):
    """input_events CSV를 결측·빈 문자열을 구분해서 읽는다.

    csv.DictWriter는 빈 문자열(inserted_text="", 순수 백스페이스 탭)과
    실제 결측(hover_to_commit_ms 없음 등)을 똑같이 빈 필드로 쓴다. pandas
    기본 read_csv는 둘 다 NaN으로 뭉개버려서 inserted_text=""가 NaN이 되면
    replay_deltas()가 "빈 문자열을 넣었다"와 "이 컬럼 자체가 없었다"를
    구분 못 하게 된다(이번 경우는 결과가 같지만, 다른 파생 지표에서 조용히
    틀린 값을 낼 수 있어 명시적으로 분리한다).

    그래서 keep_default_na=False로 전체를 문자열로 읽어 빈 문자열을
    보존한 뒤, 숫자 컬럼만 선택적으로 pd.to_numeric(errors="coerce")로
    변환한다(빈 문자열 -> NaN, 그 외는 그대로 숫자로 오독하지 않음).
    """
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )

    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].replace("", pd.NA), errors="coerce"
            )

    if "is_backspace" in df.columns:
        df["is_backspace"] = df["is_backspace"] == "True"

    return df


def sorted_run_events(df, run_id):
    """대상 run_id의 행만 골라 event_seq 기준으로 정렬한다.

    event_seq는 InputEventLogger 인스턴스 내부에서 0부터 엄격히 증가하는
    카운터라(로거 생성 1회 = run 1회), ts_ms(모노토닉 시계 해상도 약 15ms로
    동일 값이 나올 수 있음)보다 신뢰할 수 있는 정렬 기준이다.
    """
    run_df = df[df["run_id"] == run_id]
    if "event_seq" in run_df.columns:
        run_df = run_df.sort_values("event_seq", kind="stable")
    return run_df.reset_index(drop=True)


def derive_run_summary(df, run_id):
    """run_id 전체(모드 무관)의 tap_count/backspace_count/replay 결과와
    taps_per_char(NEW-02)를 낸다.

    taps_per_char는 run x keyboard_layout 단위에서만 정의된다 - "확정
    문자 1개를 만드는 데 탭이 몇 번 필요했는가"는 그 문자를 만든 탭들의
    입력 모드(dwell/mouth)가 섞여 있어도(예: dwell 탭이 만든 초성에
    mouth 탭이 모음을 눌러 완성) 성립하는 run 전체 단위 지표이지, 특정
    모드 하나에 귀속되는 값이 아니다. keyboard_layout은 run 하나당 항상
    고정이므로(CLAUDE.md "레이아웃은 런타임 전환 불가") run 수준 값이
    곧 run x keyboard_layout 수준 값과 같다.
    """
    run_df = sorted_run_events(df, run_id)

    final_text = replay_deltas(
        zip(run_df["deleted_count"], run_df["inserted_text"])
    )
    tap_count = int(len(run_df))
    backspace_count = int(run_df["is_backspace"].sum()) if tap_count else 0
    char_count = len(final_text)
    keyboard_layout = (
        run_df["keyboard_layout"].iloc[0] if tap_count else None
    )

    return {
        "run_id": run_id,
        "keyboard_layout": keyboard_layout,
        "tap_count": tap_count,
        "backspace_count": backspace_count,
        "final_text": final_text,
        "char_count": char_count,
        # NEW-02: taps_per_char, 백스페이스 포함/제외 두 버전
        "taps_per_char_incl_backspace": (
            tap_count / char_count if char_count else None
        ),
        "taps_per_char_excl_backspace": (
            (tap_count - backspace_count) / char_count
            if char_count else None
        ),
    }


def derive_group_metrics(df, run_id):
    """run_id x input_mode x keyboard_layout 단위 파생 지표 목록을 낸다.

    taps_per_char(문자당 탭 수)는 이 함수가 아니라 derive_run_summary()가
    낸다 - 확정 문자(char_count)는 tap 시퀀스 전체를 replay한 결과이지
    특정 입력 모드 하나에 귀속될 수 없기 때문이다(derive_run_summary
    docstring 참고). 대신 이 함수는 "이 모드가 전체 탭 중 몇 %를
    차지했는가"(tap_share_pct)를 제공한다 - 모드 간 상대적 사용 비중을
    보는 지표로, taps_per_char처럼 문자 단위 분모를 필요로 하지 않는다.
    """
    run_df = sorted_run_events(df, run_id)
    summary = derive_run_summary(df, run_id)
    total_tap_count = summary["tap_count"]

    rows = []
    if run_df.empty:
        return rows

    for (input_mode, keyboard_layout), group in run_df.groupby(
        ["input_mode", "keyboard_layout"], sort=False
    ):
        tap_count = int(len(group))
        backspace_count = int(group["is_backspace"].sum())
        hover = group["hover_to_commit_ms"].dropna()

        rows.append({
            "run_id": run_id,
            "input_mode": input_mode,
            "keyboard_layout": keyboard_layout,
            "tap_count": tap_count,
            "backspace_count": backspace_count,
            "tap_share_pct": (
                tap_count / total_tap_count * 100
                if total_tap_count else None
            ),
            # ACC-02: 탭 기준 오류율
            "error_rate": (
                backspace_count / tap_count if tap_count else None
            ),
            "hover_to_commit_ms_mean": (
                float(hover.mean()) if len(hover) else None
            ),
            "hover_to_commit_ms_median": (
                float(hover.median()) if len(hover) else None
            ),
        })

    return rows


def default_input_events_path(results_dir="gaze_accuracy_results"):
    import os
    return os.path.join(
        results_dir, f"input_events_v{InputEventLogger.SCHEMA_VERSION}.csv"
    )
