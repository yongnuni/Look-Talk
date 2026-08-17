from src.keyboard import get_button_center, DISPLAY_LABELS
from src.common import clock


def log_input_tap(
    input_event_logger,
    frame_id,
    input_mode,
    keyboard_layout,
    key_id,
    button_list,
    target_char,
    hover_start_ts_ms,
    cursor_x,
    cursor_y,
    deleted_count,
    inserted_text,
    trigger_signal,
):
    """dwell/mouth 훅 공용 - tap_commit 이벤트 한 건을 InputEventLogger에 넘긴다.

    button_list는 반드시 process_key() 호출 "이전" 버튼 목록이어야 한다 -
    한/영 전환처럼 process_key()가 buttonList 자체를 새로 만들면, 방금 누른
    key_id(예: "ㅂ")가 새 버튼 목록에는 없어 key_center를 못 찾기 때문이다.
    target_char도 마찬가지로 호출자가 process_key() 호출 "이전"에 계산해
    넘겨야 한다 - 이 탭이 실제로 노렸던 목표 문자를 남기려는 것이지,
    이 탭 이후 다음에 쳐야 할 문자를 남기려는 게 아니다.
    """
    key_center = get_button_center(button_list, key_id)
    key_center_x, key_center_y = key_center if key_center else (None, None)

    hover_to_commit_ms = (
        clock.now_ms() - hover_start_ts_ms
        if hover_start_ts_ms is not None
        else None
    )

    input_event_logger.log_tap_commit(
        frame_id=frame_id,
        input_mode=input_mode,
        keyboard_layout=keyboard_layout,
        key_id=key_id,
        key_label=DISPLAY_LABELS.get(key_id, key_id),
        is_backspace=(key_id == "Del"),
        deleted_count=deleted_count,
        inserted_text=inserted_text,
        hover_to_commit_ms=hover_to_commit_ms,
        cursor_x=cursor_x,
        cursor_y=cursor_y,
        key_center_x=key_center_x,
        key_center_y=key_center_y,
        target_char=target_char,
        trigger_signal=trigger_signal,
    )
