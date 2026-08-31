"""BlinkEventLogger: 완결된 깜빡임 이벤트(BlinkEvent)를 1건당 1행으로 기록하는 로거.

src/metrics/input_event_logger.py와 같은 설계 원칙을 따른다:
    - main.py의 공통 루프에서만 호출된다.
    - 매 이벤트마다 파일 I/O를 하면 카메라 루프가 느려지므로, 메모리에 버퍼링한
      뒤 일정 개수마다 + 종료 시 flush한다.
    - 로깅 실패가 입력 파이프라인을 막으면 안 된다 - 모든 쓰기는 방어적으로
      처리하고, 예외를 밖으로 던지지 않는다.

BlinkDetector는 깜빡임이 완결된 프레임에서만 BlinkEvent를 반환하므로
(src/tracking/blink.py), 이 로거는 그 반환값이 None이 아닌 프레임에서만
log_event()로 호출된다. 눈 깜빡임 캘리브레이션 완료 시 main.py가
blink_detector를 개인화된 임계값으로 재인스턴스화하므로, 이 로거는 특정
BlinkDetector 인스턴스를 생성 시점에 캡처하지 않고 매 프레임의 반환값만
전달받는다 - 재인스턴스화 이후에도 그대로 계속 기록된다.
"""

import os

from src.common import clock


class BlinkClosureEarTracker:
    """감음(is_closed) 구간 동안 관측된 EAR의 최솟값을 추적한다.

    BlinkDetector(src/tracking/blink.py)는 눈을 다시 뜬 프레임
    (ear > open_threshold)에서만 NATURAL/INTENTIONAL 이벤트를 반환한다.
    그 프레임의 ear를 그대로 로깅하면 "감았을 때"가 아니라 "막 뜬 순간"의
    값이 기록되므로, 매 프레임 observe()를 호출해 감음 구간의 최솟값을
    유지하고 그 값을 이벤트와 함께 기록한다.
    """

    def __init__(self):
        self._min_ear = None

    def observe(self, is_closed, ear):
        """이번 프레임 결과를 반영하고, 이번 프레임에 로깅할 ear 값을 반환한다.

        계속 감겨 있는 동안(LONG_CLOSURE로 이어지는 경우 포함)은 누적
        최솟값을 갱신하며 그 값을 반환한다. 눈을 다시 뜬 프레임에서는
        직전까지의 감음 구간 최솟값을 반환한 뒤 다음 구간을 위해
        리셋한다.
        """
        if is_closed:
            self._min_ear = (
                ear if self._min_ear is None else min(self._min_ear, ear)
            )
            return self._min_ear

        value = self._min_ear if self._min_ear is not None else ear
        self._min_ear = None
        return value


class BlinkEventLogger:

    SCHEMA_VERSION = "1.0"

    FIELDNAMES = [
        "run_id",
        "ts_ms",
        "ear_at_close",
        "kind",
        "duration_ms",
    ]

    def __init__(
        self,
        path=None,
        flush_every=10,
        enabled=True,
        run_id=None,
    ):
        self.run_id = run_id
        self.path = path or os.path.join(
            "gaze_accuracy_results",
            f"blink_events_v{self.SCHEMA_VERSION}.csv",
        )
        self.flush_every = max(1, flush_every)
        self.enabled = enabled

        self._buffer = []
        self._write_failed = False
        self._warned_once = False
        self._dropped_event_count = 0

    def log_event(self, blink_event, ear_at_close):
        """완결된 BlinkEvent 1건을 기록한다.

        ear_at_close는 이벤트가 반환된 프레임의 EAR이 아니라 **감음 구간
        동안 관측된 EAR의 최솟값**이어야 한다 - BlinkDetector는 눈을 다시 뜬
        프레임(ear > open_threshold)에서만 NATURAL/INTENTIONAL 이벤트를
        반환하므로, 호출 프레임의 ear를 그대로 넘기면 "감았을 때"가 아니라
        "막 뜬 순간"의 값이 기록된다(호출부 main.py에서 구간 최솟값을 직접
        추적해 넘긴다).
        """
        if not self.enabled or blink_event is None:
            return

        if self._write_failed:
            self._dropped_event_count += 1
            return

        try:
            row = {
                "run_id": self.run_id,
                "ts_ms": clock.now_ms(),
                "ear_at_close": ear_at_close,
                "kind": blink_event.kind.name,
                "duration_ms": round(blink_event.duration * 1000.0, 1),
            }
        except Exception as e:
            self._dropped_event_count += 1
            self._warn_once(f"이벤트 기록 준비 실패(무시하고 계속 진행): {e}")
            return

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
                f"CSV 쓰기 실패, 이후 blink_events 기록을 비활성화합니다: {e}"
            )

    def _warn_once(self, message):
        if not self._warned_once:
            print(f"[blink_event_logger] {message}")
            self._warned_once = True

    def close(self):
        self.flush()

        if self._dropped_event_count > 0:
            print(
                f"[blink_event_logger] 종료 - 기록 실패로 유실된 이벤트 "
                f"{self._dropped_event_count}건 (blink_events.csv에 반영되지 않음)"
            )
