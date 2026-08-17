import os

from src.common import clock
from src.metrics.csv_export import append_rows


def export_targeting_results(run_id, keyboard_layout, targeting_runner, aborted):
    """10회 원형 타겟팅 테스트 결과를 타깃 1개당 1행으로 저장한다.

    지금까지는 get_results()가 화면 렌더링에만 쓰이고 앱 종료 시 그대로
    소실됐다(docs/current_state_report.md 1-2절). TargetingTestRunner
    자체는 팀원 코드라 손대지 않고, 여기(main.py)에서 완료·중도 종료
    시점에 결과만 꺼내 CSV로 내보낸다.

    input_mode는 TargetAttempt.reason("dwell"/"selection_hit"/
    "selection_miss"/"timeout")에서 역산한다 — 러너가 이미 구분해 두고
    있어 별도 판정 로직을 추가하지 않는다. target_x/y, cursor(=selected)_x/y도
    TargetAttempt에 이미 있어 함께 기록한다.

    reason → input_mode 매핑 근거 (tests/targeting_test_runner.py 기준,
    팀원 코드라 확인만 하고 직접 수정하지 않음):
    - "dwell": update_dwell()의 성공 분기(targeting_test_runner.py:280-287)
      에서만 설정된다. gaze가 dwell_sec 이상 연속으로 타겟 내부에 머물렀을
      때만 도달하는 경로라 mouth_click과 무관 — "dwell" 트리거로 확정.
    - "selection_hit"/"selection_miss": register_selection()
      (targeting_test_runner.py:313-328)에서만 설정되고, 이 메서드는
      main.py에서 mouth.update()의 반환값 mouth_click이 True일 때만
      호출된다(위 mouth_click, mar = mouth.update(...) 및 아래
      register_selection() 호출부 참고) — "mouth"(입벌림) 트리거로 확정.
    - "timeout": update_dwell()의 타임아웃 분기(targeting_test_runner.py:
      258-265)에서 설정된다. update_dwell()과 mouth_click 판정(바로 위
      if/else 블록)은 targeting_mode 활성 중 매 프레임 mouth_mode 값과
      무관하게 함께 실행된다 — 즉 한 trial 안에서 dwell·mouth 두 트리거가
      항상 동시에 살아있어, timeout은 "둘 다 시간 안에 성공하지 못함"만
      의미할 뿐 사용자가 어느 쪽을 시도했는지 구분할 근거가 없다. 추측으로
      채우지 않고 input_mode를 비워 기록한다(None).
    """
    if not targeting_runner.attempts:
        return

    fieldnames = [
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
    ]

    ts_ms = clock.now_ms()
    rows = []

    for attempt in targeting_runner.attempts:
        if attempt.reason == "dwell":
            input_mode = "dwell"  # update_dwell() 성공 분기 전용 — 근거는 위 함수 docstring
        elif attempt.reason in ("selection_hit", "selection_miss"):
            input_mode = "mouth"  # register_selection()은 mouth_click=True일 때만 호출됨
        else:
            input_mode = None  # timeout: dwell·mouth 트리거가 동시에 살아있어 구분 불가 — 추측 금지

        rows.append({
            "run_id": run_id,
            "keyboard_layout": keyboard_layout,
            "ts_ms": ts_ms,
            "target_index": attempt.trial,
            "success": attempt.success,
            "reaction_time_sec": round(attempt.reaction_time_sec, 3),
            "input_mode": input_mode,
            "timeout": attempt.reason == "timeout",
            "target_x": attempt.target_x,
            "target_y": attempt.target_y,
            "cursor_x": attempt.selected_x,
            "cursor_y": attempt.selected_y,
            "aborted": aborted,
        })

    os.makedirs("gaze_accuracy_results", exist_ok=True)

    append_rows(
        os.path.join("gaze_accuracy_results", "targeting_results_v1.0.csv"),
        fieldnames,
        rows,
    )
