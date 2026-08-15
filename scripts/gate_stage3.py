"""3단계(원시 이벤트 -> 파생 지표) 관문 스크립트.

목적(관문 이력): derive_input.py가 input_events로부터 파생한
tap_count/backspace_count가 TestRunner 집계와 일치하는지 대조해, 3단계
관문용으로 sessions에 임시로 실었던 keystrokes/backspace_count 컬럼(v1.7)과
대조했다. 2026-08-16 실행 결과 keystrokes 16/16, backspace_count 2/2 모두
일치해 관문을 통과했고, 사용자 승인을 받아 TestRunner의 집계 로직과 sessions의
해당 컬럼(v1.8에서 제거, 아래 GATE_A_HISTORICAL_NOTE)을 제거했다. 그 이후로는
derive_input이 이 값들의 유일한 원천이라 이 스크립트가 라이브로 대조할 대상이
없다 - 대조 A는 그 이력을 보여주는 안내문으로 남는다.

사용법 (프로젝트 루트에서, main.py를 캘리 -> t -> 문장 완주 -> q 순서로 최소
1회 실행해 CSV를 만든 뒤):
    python -m scripts.gate_stage3
    python -m scripts.gate_stage3 --run-id <run_id>

대조 A: (제거됨, 이력 안내만 출력 - 위 참고)
대조 B: replay 결과 vs 목표 문장. sessions.export_reason이 sentence_completed인
        실행만 검증 가능하다 - 그 경우 derive_input.find_sentence_completion()으로
        해당 run의 탭을 순서대로 replay하며 tests/test_sentences.TEST_SENTENCES
        중 하나와 처음 일치하는 시점을 찾는다(목표 문장 자체는 어디에도
        저장되지 않으므로 - sessions/TestRunner 둘 다 미보존 - "완주 시 반드시
        TEST_SENTENCES 중 하나와 같아진다"는 사실로 대신 검증. 완주 이후 같은
        run에서 계속 타이핑해도 첫 일치 시점만 취하므로 섞이지 않는다).
tap>char: QWERTY 한글 조합 구간은 자모 탭 수가 완성 문자 수보다 항상 많아야
        한다(자모 1개 = 탭 1개, 완성 문자 1개 = 자모 2~3개). 천지인은 데이터가
        있으면 같은 방식으로, 없으면 "데이터 없음"으로 표시한다.
첫 조인 분석: 대상 run의 모드별(dwell/mouth) tap 수, hover_to_commit_ms 분포,
        tap 점유율(%)과 run 수준 taps_per_char를 표로 출력한다.
"""

import argparse
import os
import sys

from src.metrics.collector import MetricsCollector
from src.metrics.session_logger import SessionLogger
from src.metrics import derive_input
from scripts.verify_stage1 import _load_csv, select_target_run_id
from tests.test_sentences import TEST_SENTENCES

# 3단계 관문이 실제로 통과한 이력(사람이 검증한 결과 - 스크립트가 재현하지
# 않는다. sessions.keystrokes/backspace_count 컬럼 자체가 v1.8에서 제거돼
# 더 이상 라이브 대조가 불가능하므로, 이 상수를 갱신하는 것도 사람이 한다).
GATE_A_HISTORICAL_NOTE = (
    "2026-08-16, keystrokes 16/16 - backspace_count 2/2 일치 확인 "
    "(사용자 승인 후 TestRunner 집계 및 sessions 컬럼 제거, schema v1.7 -> v1.8)"
)


def _sessions_row(sessions_df, run_id):
    if sessions_df is None or "run_id" not in sessions_df.columns:
        return None
    matches = sessions_df[sessions_df["run_id"] == run_id]
    if matches.empty:
        return None
    # sessions는 run_id당 1행이 원칙이나(안전망 저장), 여러 번 append된
    # 이상 상태를 대비해 마지막 행을 쓴다(가장 최근 기록).
    return matches.iloc[-1]


def check_compare_a():
    """대조 A: (제거됨) TestRunner 집계 vs derive_input 파생값.

    sessions.keystrokes/backspace_count는 3단계 관문 통과 후 제거됐다(스크립트
    상단 docstring 참고) - 더 이상 sessions에 저장되지 않으므로 라이브 대조
    대상이 없다. 항상 [정보]만 출력하고 스킵 처리한다(실패로 잡지 않는다) -
    "비교 대상 자체가 의도적으로 없어졌다"는 상태가 실패일 이유는 없기 때문이다.
    """
    print("\n=== 대조 A: TestRunner 집계 vs derive_input 파생값 ===")
    print("  [정보] TestRunner의 keystrokes/backspace_count 집계 로직과 "
          "sessions의 해당 컬럼은 3단계 검증 통과 후 제거됨 - 대조 대상 없음")
    print(f"  3단계 검증 당시 이력: {GATE_A_HISTORICAL_NOTE}")
    return None


def check_compare_b(sessions_row, events_df, run_id):
    """대조 B: replay 결과 vs 목표 문장(TEST_SENTENCES 멤버십으로 대신 검증).

    목표 문장 자체(TestRunner.target_text)는 어디에도 저장되지 않는다 - 그래서
    "완주 시 최종 문자열이 TEST_SENTENCES 중 하나와 정확히 같아야 한다"는
    사실로 대신 검증한다. sentence_completed가 아니면 애초에 완주하지
    않았으므로 검증 대상이 아니다.
    """
    print("\n=== 대조 B: replay 결과 vs 목표 문장 ===")

    if sessions_row is None:
        print("  [스킵] 대상 run_id의 sessions 행이 없음")
        return None

    export_reason = sessions_row.get("export_reason")

    if export_reason != MetricsCollector.EXPORT_REASON_SENTENCE_COMPLETED:
        print(f"  [정보] export_reason={export_reason!r} - 문장을 끝까지 "
              "완주한 실행이 아니라 대조 불가")
        return None

    match = derive_input.find_sentence_completion(
        events_df, run_id, TEST_SENTENCES
    )

    if match is None:
        print("  [실패] 이 run의 input_events 어느 시점에도 TEST_SENTENCES와 "
              "일치하는 상태가 없음 - replay 로직 또는 완주 판정 로직 확인 필요")
        print(f"         TEST_SENTENCES={TEST_SENTENCES}")
        return False

    print(f"  [통과] {match['tap_count']}번째 탭에서 목표 문장 완성 확인: "
          f"{match['text']!r}")
    return True


def check_tap_greater_than_char(events_df, run_id):
    """QWERTY/천지인 한글 조합 구간에서 tap_count > char_count가 성립하는지."""
    print("\n=== tap_count > char_count 검증 (한글 조합 구간) ===")

    run_df = derive_input.sorted_run_events(events_df, run_id)

    if run_df.empty:
        print("  [스킵] 이 run에 input_events 없음")
        return None

    ok = True
    for layout in ("qwerty", "cheonjiin"):
        layout_df = run_df[run_df["keyboard_layout"] == layout]

        if layout_df.empty:
            print(f"  [정보] {layout}: 데이터 없음")
            continue

        tap_count = len(layout_df)
        final_text = derive_input.replay_deltas(
            zip(layout_df["deleted_count"], layout_df["inserted_text"])
        )
        char_count = len(final_text)

        if tap_count > char_count:
            print(f"  [통과] {layout}: tap_count({tap_count}) > "
                  f"char_count({char_count})")
        else:
            print(f"  [실패] {layout}: tap_count({tap_count}) <= "
                  f"char_count({char_count}) - 한글 자모 조합이라면 "
                  "성립해야 함")
            ok = False

    return ok


def print_join_table(events_df, run_id):
    """모드별 tap 수 / hover_to_commit_ms 분포 / tap 점유율 표 + run 수준
    taps_per_char를 출력한다."""
    print("\n=== 첫 조인 분석: run_id별 모드 x 지표 ===")

    summary = derive_input.derive_run_summary(events_df, run_id)
    rows = derive_input.derive_group_metrics(events_df, run_id)

    if not rows:
        print("  이 run에 input_events 없음")
        return

    def fmt(value, digits=1):
        return "-" if value is None else f"{value:.{digits}f}"

    print(f"  run 수준: keyboard_layout={summary['keyboard_layout']}  "
          f"tap_count={summary['tap_count']}  char_count={summary['char_count']}  "
          f"taps_per_char(+bs)={fmt(summary['taps_per_char_incl_backspace'], 2)}  "
          f"taps_per_char(-bs)={fmt(summary['taps_per_char_excl_backspace'], 2)}")
    print()

    header = (
        "  mode".ljust(10) + "layout".ljust(12) + "tap".ljust(8)
        + "backspace".ljust(12) + "tap_share_pct".ljust(16)
        + "hover_mean_ms".ljust(16) + "hover_median_ms"
    )
    print(header)

    for row in rows:
        line = (
            f"  {row['input_mode']}".ljust(10)
            + str(row["keyboard_layout"]).ljust(12)
            + str(row["tap_count"]).ljust(8)
            + str(row["backspace_count"]).ljust(12)
            + fmt(row["tap_share_pct"]).ljust(16)
            + fmt(row["hover_to_commit_ms_mean"]).ljust(16)
            + fmt(row["hover_to_commit_ms_median"])
        )
        print(line)


def main():
    parser = argparse.ArgumentParser(description="3단계 검증: derive_input 대조")
    parser.add_argument("--results-dir", default="gaze_accuracy_results")
    parser.add_argument(
        "--run-id", default=None,
        help="검사 대상 run_id 직접 지정 (기본: mapper_session_log 최신 run_id)"
    )
    args = parser.parse_args()

    sessions_path = os.path.join(
        args.results_dir, f"sessions_v{MetricsCollector.SCHEMA_VERSION}.csv"
    )
    session_log_path = os.path.join(
        args.results_dir, f"mapper_session_log_v{SessionLogger.SCHEMA_VERSION}.csv"
    )
    input_events_path = derive_input.default_input_events_path(args.results_dir)

    print("=" * 60)
    print("입력 기록 검증: CSV 로딩")
    print("=" * 60)

    sessions_df = _load_csv(sessions_path, "sessions", required=True)
    session_log_df = _load_csv(session_log_path, "mapper_session_log")

    if not os.path.isfile(input_events_path):
        print(f"  [실패] input_events 파일 없음: {input_events_path}")
        events_df = None
    else:
        events_df = derive_input.load_input_events(input_events_path)
        print(f"  [로드] input_events: {input_events_path} ({len(events_df)}행)")

    target_run_id, selection_note = select_target_run_id(
        session_log_df, args.run_id
    )

    if target_run_id is None:
        print(f"\n[실패] 대상 run_id를 정할 수 없음: {selection_note}")
        sys.exit(1)

    print(f"\n대상 run_id: {target_run_id} (선정 기준: {selection_note})")

    if sessions_df is None or events_df is None:
        print("\n[결론] 필수 파일 누락으로 검증 판정 불가.")
        sys.exit(1)

    sessions_row = _sessions_row(sessions_df, target_run_id)

    results = {
        "compare_a_testrunner_vs_derive": check_compare_a(),
        "compare_b_replay_vs_target_sentence": check_compare_b(
            sessions_row, events_df, target_run_id
        ),
        "tap_greater_than_char": check_tap_greater_than_char(events_df, target_run_id),
    }

    print_join_table(events_df, target_run_id)

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    for name, result in results.items():
        status = "통과" if result is True else ("스킵" if result is None else "실패")
        print(f"  {name}: {status}")

    if any(r is False for r in results.values()):
        print("\n[결론] 일부 검증 실패 - 위 로그에서 원인을 확인하고 사용자에게 보고하세요.")
        sys.exit(1)

    print("\n[결론] 전체 검증 통과. 스킵 항목이 있다면 그 이유(문장 미완주 "
          "등)를 확인하세요.")


if __name__ == "__main__":
    main()
