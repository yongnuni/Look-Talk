"""
SessionLogger: calibrated/no_calibration 두 모드가 공통으로 쓰는 프레임 단위
로거.

원칙(설계 지시사항):
    - Mapper 내부에 있지 않다 — main.py의 공통 루프에서만 호출된다.
    - mode/mapper metadata를 항상 같이 기록해 향후 여러 Mapper를 같은
      스키마로 비교할 수 있게 한다.
    - calibration 관련 필드는 no_calibration에서 자연스럽게 빈 값이 된다
      (여기서 강제로 채우지 않는다).
    - 로깅 실패가 입력 파이프라인을 막으면 안 된다 — 모든 쓰기는 방어적으로
      처리하고, 실패해도 예외를 밖으로 던지지 않는다.
    - 매 프레임 파일 I/O를 하면 카메라 루프가 느려지므로, 메모리에 버퍼링한
      뒤 일정 개수마다 + 종료 시 flush한다.

이 로거는 9점 정확도 테스트 전용인 MetricsCollector(src/metrics/collector.py)
를 대체하지 않는다. 그 클래스는 t키 테스트에서 지금처럼 그대로 쓰고, 이
로거는 "일반 사용 중 매 프레임"을 기록하는 별도의 목적을 가진다.
"""

import os
import time
import uuid


class SessionLogger:

    FIELDNAMES = [
        "session_id",
        "timestamp",
        "mode",
        "active_method",
        "screen_w",
        "screen_h",
        "raw_iris_x",
        "raw_iris_y",
        "mapped_sx",
        "mapped_sy",
        "mapping_valid",
        "gaze_x",
        "gaze_y",
        "hovered_key",
        "dwell_ratio",
        "clicked_key",
        "input_text_len",
        "mapper_metadata",
    ]

    def __init__(
        self,
        mode,
        mapper_metadata=None,
        screen_w=None,
        screen_h=None,
        path=os.path.join("gaze_accuracy_results", "mapper_session_log_v1.0.csv"),
        flush_every=60,
        enabled=True,
    ):
        self.session_id = str(uuid.uuid4())
        self.mode = mode
        self.mapper_metadata = mapper_metadata or {}
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.path = path
        self.flush_every = max(1, flush_every)
        self.enabled = enabled

        self._buffer = []
        self._write_failed = False

    def log_frame(
        self,
        active_method,
        raw_iris_x,
        raw_iris_y,
        mapped_sx,
        mapped_sy,
        mapping_valid,
        gaze_x,
        gaze_y,
        hovered_key,
        dwell_ratio,
        clicked_key,
        input_text_len,
    ):
        if not self.enabled or self._write_failed:
            return

        try:
            self._buffer.append({
                "session_id": self.session_id,
                "timestamp": time.time(),
                "mode": self.mode,
                "active_method": active_method,
                "screen_w": self.screen_w,
                "screen_h": self.screen_h,
                "raw_iris_x": raw_iris_x,
                "raw_iris_y": raw_iris_y,
                "mapped_sx": mapped_sx,
                "mapped_sy": mapped_sy,
                "mapping_valid": mapping_valid,
                "gaze_x": gaze_x,
                "gaze_y": gaze_y,
                "hovered_key": hovered_key,
                "dwell_ratio": round(dwell_ratio, 4) if dwell_ratio is not None else None,
                "clicked_key": clicked_key,
                "input_text_len": input_text_len,
                "mapper_metadata": str(self.mapper_metadata),
            })

            if len(self._buffer) >= self.flush_every:
                self.flush()

        except Exception as e:
            # 로깅 실패로 입력 파이프라인을 막지 않는다.
            print(f"[session_logger] 프레임 기록 준비 실패(무시하고 계속 진행): {e}")

    def flush(self):
        if not self.enabled or not self._buffer:
            return

        try:
            from src.metrics.csv_export import append_rows

            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            append_rows(self.path, self.FIELDNAMES, self._buffer)
            self._buffer = []

        except Exception as e:
            # 디스크 오류 등으로 반복 실패하는 걸 막기 위해 이후 로깅은 끈다.
            print(f"[session_logger] CSV 쓰기 실패, 이후 로깅을 비활성화합니다: {e}")
            self._write_failed = True
            self._buffer = []

    def close(self):
        self.flush()
