"""시선 정확도 지표 수집 모듈.

기존 MVP 코드와 분리된 독립 모듈이며, main.py에 훅으로 연결된다.
'목표 키의 화면 좌표 = 정답 좌표'로 두고, 사용자가 그 키를 노리는 동안의
시선 예측 좌표를 매 프레임 쌓아 타깃별 오차/표준편차/추적실패율을 계산한다.

수집 단위
- 세션 1개  -> sessions.csv 한 행 (메타데이터)
- 타깃 N개  -> gaze_accuracy.csv N행 (지표 본체)
두 파일은 session_id로 연결된다.
"""

import os
import csv
import math
import uuid
import statistics
from datetime import datetime, timezone


class MetricsCollector:

    SCHEMA_VERSION = "1.1"

    def __init__(self, user_id="anonymous", dev_version="v0.1-raw", px_per_cm=None):
    # 세션 단위 메타데이터 (sessions.csv 한 행)
        self.session_id = str(uuid.uuid4())
        self.user_id = user_id
        self.dev_version = dev_version
        self.px_per_cm = px_per_cm
        self.start_timestamp = datetime.now(timezone.utc).isoformat()
        self.end_timestamp = None

        self.target_rows = []
        self._current = None

    # ── 측정 상태 조회 (main.py가 내부 변수 직접 참조하지 않도록) ──

    def is_measuring(self):
        """현재 타깃이 열려 있는지(=샘플을 받을 상태인지) 반환."""
        return self._current is not None

    # ── 타깃 시작 ──────────────────────────────────────────

    def start_target(self, target_index, target_x_px, target_y_px):
        """새 타깃 응시 시작. 이전 버퍼를 비우고 정답 좌표를 기록한다."""
        self._current = {
            "target_index": target_index,
            "target_x_px": target_x_px,
            "target_y_px": target_y_px,
            "pred_xs": [],
            "pred_ys": [],
            "iris_xs": [],
            "iris_ys": [],
            "valid_count": 0,   # 추적 성공 프레임 (오차 계산에 쓰임)
            "total_count": 0,   # 전체 프레임 (성공 + 실패)
        }

    # ── 매 프레임 샘플 ─────────────────────────────────────

    def add_sample(self, gaze_x, gaze_y, iris_x, iris_y):
        """매 프레임 호출. 현재 타깃 버퍼에 예측·홍채 좌표를 쌓는다.

        gaze_x/y: GazePipeline 최종 좌표 (사용자가 실제 경험하는 시선 위치)
        iris_x/y: get_avg_iris() 홍채 중심 (STB-04 신호 안정성용)
        """
        if self._current is None:
            return

        # 이 타깃에서 들어온 전체 프레임 수 (분모)
        self._current["total_count"] += 1

        # 추적 실패 프레임은 좌표를 쌓지 않고 카운트만 한다.
        # main.py에서 얼굴 미검출/저신뢰 시 gaze_x/y = -1 로 들어옴.
        if gaze_x < 0 or gaze_y < 0:
            return

        self._current["pred_xs"].append(gaze_x)
        self._current["pred_ys"].append(gaze_y)
        self._current["iris_xs"].append(iris_x)
        self._current["iris_ys"].append(iris_y)
        self._current["valid_count"] += 1

    # ── 타깃 종료 (지표 계산) ──────────────────────────────

    def end_target(self):
        """현재 타깃 응시 종료. 쌓인 좌표로 지표를 계산해 target_rows에 한 행 추가."""
        if self._current is None:
            return

        c = self._current
        total = c["total_count"]
        valid = c["valid_count"]

        # STB-02 Dropout Rate
        dropout_rate = (total - valid) / total if total > 0 else None

        # 유효 프레임이 하나도 없으면 오차 계산 불가 → 실패 행으로 기록
        if valid == 0:
            self.target_rows.append({
                "session_id": self.session_id,
                "target_index": c["target_index"],
                "target_x_px": c["target_x_px"],
                "target_y_px": c["target_y_px"],
                "pred_x_px": None,
                "pred_y_px": None,
                "euclidean_error_px": None,
                "euclidean_error_cm": None,
                "gaze_std_x_px": None,
                "gaze_std_y_px": None,
                "iris_std_x_px": None,
                "iris_std_y_px": None,
                "dropout_rate": round(dropout_rate, 4) if dropout_rate is not None else None,
                "sample_count": 0,
            })
            self._current = None
            return

        # 예측 좌표 대표값: 유효 프레임 평균
        pred_x = statistics.mean(c["pred_xs"])
        pred_y = statistics.mean(c["pred_ys"])

        # ACC-05 Euclidean Error
        euclidean_error_px = math.hypot(pred_x - c["target_x_px"],
                                        pred_y - c["target_y_px"])

        # STB-03 / STB-04 표준편차 (표본 2개 이상일 때만 정의됨)
        gaze_std_x = statistics.stdev(c["pred_xs"]) if valid > 1 else 0.0
        gaze_std_y = statistics.stdev(c["pred_ys"]) if valid > 1 else 0.0
        iris_std_x = statistics.stdev(c["iris_xs"]) if valid > 1 else 0.0
        iris_std_y = statistics.stdev(c["iris_ys"]) if valid > 1 else 0.0

        self.target_rows.append({
            "session_id": self.session_id,
            "target_index": c["target_index"],
            "target_x_px": c["target_x_px"],
            "target_y_px": c["target_y_px"],
            "pred_x_px": round(pred_x, 2),
            "pred_y_px": round(pred_y, 2),
            "euclidean_error_px": round(euclidean_error_px, 2),
            "euclidean_error_cm": self._to_cm(euclidean_error_px),
            "gaze_std_x_px": round(gaze_std_x, 2),
            "gaze_std_y_px": round(gaze_std_y, 2),
            "iris_std_x_px": round(iris_std_x, 2),
            "iris_std_y_px": round(iris_std_y, 2),
            "dropout_rate": round(dropout_rate, 4),
            "sample_count": valid,
        })
        self._current = None

    # ── 세션 종료 / 내보내기 ───────────────────────────────

    def end_session(self):
        """세션 종료 시각 기록. export 전에 호출."""
        self.end_timestamp = datetime.now(timezone.utc).isoformat()

    def export_csv(self, sessions_path="sessions.csv",
                   accuracy_path="gaze_accuracy.csv"):
        """메모리에 쌓인 지표를 두 CSV 파일에 append 모드로 내보낸다."""
        if self.end_timestamp is None:
            self.end_session()

        session_fields = [
            "session_id", "user_id", "dev_version",
            "start_timestamp", "end_timestamp",
            "session_duration_total_ms", "px_per_cm", "schema_version",
        ]
        session_row = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "dev_version": self.dev_version,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "session_duration_total_ms": self._compute_duration_ms(),
            "px_per_cm": round(self.px_per_cm, 3) if self.px_per_cm else None,
            "schema_version": self.SCHEMA_VERSION,
        }
        self._append_rows(sessions_path, session_fields, [session_row])

        accuracy_fields = [
            "session_id", "target_index",
            "target_x_px", "target_y_px",
            "pred_x_px", "pred_y_px",
            "euclidean_error_px", "euclidean_error_cm",
            "gaze_std_x_px", "gaze_std_y_px",
            "iris_std_x_px", "iris_std_y_px",
            "dropout_rate", "sample_count",
        ]
        self._append_rows(accuracy_path, accuracy_fields, self.target_rows)

    # ── 내부 헬퍼 ──────────────────────────────────────────

    def _compute_duration_ms(self):
        start = datetime.fromisoformat(self.start_timestamp)
        end = datetime.fromisoformat(self.end_timestamp)
        return int((end - start).total_seconds() * 1000)
    
    def _to_cm(self, error_px):
        """px 오차를 cm로 환산. px_per_cm이 없으면 None."""
        if error_px is None or self.px_per_cm is None or self.px_per_cm == 0:
            return None
        return round(error_px / self.px_per_cm, 3)

    def _append_rows(self, path, fieldnames, rows):
        file_exists = os.path.isfile(path)
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
