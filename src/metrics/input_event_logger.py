"""InputEventLogger: 키 입력을 이벤트(탭) 단위로 기록하는 로거.

SessionLogger(src/metrics/session_logger.py)와 같은 설계 원칙을 따른다:
    - main.py의 공통 루프에서만 호출된다.
    - 매 탭마다 파일 I/O를 하면 카메라 루프가 느려지므로, 메모리에 버퍼링한
      뒤 일정 개수마다 + 종료 시 flush한다.
    - 로깅 실패가 입력 파이프라인을 막으면 안 된다 - 모든 쓰기는 방어적으로
      처리하고, 예외를 밖으로 던지지 않는다.

다른 점(1단계 sessions.csv 유실 교훈 반영): 실패를 조용히 삼키지 않는다.
기록 실패는 최초 1회만 콘솔에 경고를 출력하고, 이후 실패는 카운트만 누적해
close() 시점에 요약으로 남긴다.

이번 단계(2단계)에서는 event_type이 "tap_commit" 하나만 실제로 발행된다.
음절 확정(char_commit) 이벤트는 발행하지 않는다 - 이 코드베이스의 한글/천지인
조합은 소급 재해석·재치환이 가능해(hangul.flush_buffer, 천지인
_replace_last_vowel, 백스페이스) 탭 시점에 "이 문자는 이제 안 바뀐다"를
확정할 수 없다. 대신 tap_commit 행마다 꼬리 diff(deleted_count/inserted_text)를
남기고, char_commit은 분석 단계에서 diff 시퀀스를 재생(replay)해 사후 파생한다.
"""

import os

from src.common import clock


class InputEventLogger:

    SCHEMA_VERSION = "1.0"

    EVENT_TAP_COMMIT = "tap_commit"

    FIELDNAMES = [
        "run_id",
        "ts_ms",
        "frame_id",
        "event_seq",
        "event_type",
        "input_mode",
        "keyboard_layout",
        "key_id",
        "key_label",
        "is_backspace",
        "deleted_count",
        "inserted_text",
        "hover_to_commit_ms",
        "cursor_x",
        "cursor_y",
        "key_center_x",
        "key_center_y",
        "target_char",
        "trigger_signal",
    ]

    def __init__(
        self,
        path=None,
        flush_every=30,
        enabled=True,
        run_id=None,
    ):
        self.run_id = run_id
        self.path = path or os.path.join(
            "gaze_accuracy_results",
            f"input_events_v{self.SCHEMA_VERSION}.csv",
        )
        self.flush_every = max(1, flush_every)
        self.enabled = enabled

        self._buffer = []
        self._event_seq = 0
        self._write_failed = False
        self._warned_once = False
        self._dropped_event_count = 0

    def log_tap_commit(
        self,
        frame_id,
        input_mode,
        keyboard_layout,
        key_id,
        key_label,
        is_backspace,
        deleted_count,
        inserted_text,
        hover_to_commit_ms,
        cursor_x,
        cursor_y,
        key_center_x,
        key_center_y,
        target_char,
        trigger_signal,
    ):
        if not self.enabled:
            return

        if self._write_failed:
            self._dropped_event_count += 1
            return

        try:
            row = {
                "run_id": self.run_id,
                "ts_ms": clock.now_ms(),
                "frame_id": frame_id,
                "event_seq": self._event_seq,
                "event_type": self.EVENT_TAP_COMMIT,
                "input_mode": input_mode,
                "keyboard_layout": keyboard_layout,
                "key_id": key_id,
                "key_label": key_label,
                "is_backspace": is_backspace,
                "deleted_count": deleted_count,
                "inserted_text": inserted_text,
                "hover_to_commit_ms": (
                    round(hover_to_commit_ms, 1)
                    if hover_to_commit_ms is not None
                    else None
                ),
                "cursor_x": cursor_x,
                "cursor_y": cursor_y,
                "key_center_x": key_center_x,
                "key_center_y": key_center_y,
                "target_char": target_char,
                "trigger_signal": trigger_signal,
            }
        except Exception as e:
            self._dropped_event_count += 1
            self._warn_once(f"이벤트 기록 준비 실패(무시하고 계속 진행): {e}")
            return

        self._event_seq += 1
        self._buffer.append(row)

        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if self._write_failed or not self._buffer:
            return

        try:
            from src.metrics.csv_export import append_rows

            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            append_rows(self.path, self.FIELDNAMES, self._buffer)
            self._buffer = []

        except Exception as e:
            # 디스크 오류 등으로 반복 실패하는 걸 막기 위해 이후 로깅은 끈다.
            self._dropped_event_count += len(self._buffer)
            self._buffer = []
            self._write_failed = True
            self._warn_once(
                f"CSV 쓰기 실패, 이후 input_events 기록을 비활성화합니다: {e}"
            )

    def _warn_once(self, message):
        if not self._warned_once:
            print(f"[input_event_logger] {message}")
            self._warned_once = True

    def close(self):
        self.flush()

        if self._dropped_event_count > 0:
            print(
                f"[input_event_logger] 종료 - 기록 실패로 유실된 이벤트 "
                f"{self._dropped_event_count}건 (input_events.csv에 반영되지 않음)"
            )
