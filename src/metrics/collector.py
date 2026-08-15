"""시선 정확도 지표 수집 모듈.

기존 MVP 코드와 분리된 독립 모듈이며, main.py에 훅으로 연결된다.
'목표 키의 화면 좌표 = 정답 좌표'로 두고, 사용자가 그 키를 노리는 동안의
시선 예측 좌표를 매 프레임 쌓아 타깃별 오차/표준편차/추적실패율을 계산한다.

수집 단위
- 세션 1개  -> sessions.csv 한 행 (메타데이터)
- 타깃 N개  -> gaze_accuracy.csv N행 (지표 본체)
두 파일은 test_id로 연결된다. run_id는 앱 실행 1회를 가리키는 별도 상위 식별자로,
같은 실행에서 나온 여러 test_id/calib_id를 묶어 조인할 때 쓴다.
"""

import math
import statistics
from datetime import datetime, timezone

from src.metrics.csv_export import append_rows
from src.common import ids


class MetricsCollector:

    SCHEMA_VERSION = "1.8"

    def __init__(
            self,
            user_id="anonymous",
            dev_version="v0.1-raw",
            px_per_cm=None,
            calib_id=None,
            calib_reproj_rmse_px=None,
            use_pose_corrected=False,
            use_sqpnp_corrected=False,
            gaze_avg_window=None,
            smoothing_mode=None,
            edge_mean_reproj_error_px=None,
            center_mean_reproj_error_px=None,
            calibration_fallback_used=False,
            rejected_calib_rmse_px=None,
            applied_calib_rmse_px=None,
            run_id=None,
            keyboard_layout=None,
            config_hash=None,
            config_json=None,
            t0_utc=None,
            screen_w=None,
            screen_h=None,
            monitor_diagonal_inch=None,
            ridge_enabled=None,
            backbone_enabled=None,
    ):
    # 세션 단위 메타데이터 (sessions.csv 한 행)
        # test_id: "9점 테스트 1회" 식별자. 기존 이름은 session_id였다
        # (docs/current_state_report.md 2-4절 — 앱 실행 전체를 가리키는 run_id와
        # 혼동을 피하려고 개명했다). 발급 자체는 uuid4 그대로, 출처만
        # src.common.ids로 통일했다.
        self.test_id = ids.new_test_id()
        self.run_id = run_id
        self.user_id = user_id
        self.dev_version = dev_version
        self.px_per_cm = px_per_cm
        self.calib_id = calib_id
        self.calib_reproj_rmse_px = calib_reproj_rmse_px
        self.gaze_avg_window = gaze_avg_window
        self.smoothing_mode = smoothing_mode
        self.edge_mean_reproj_error_px = edge_mean_reproj_error_px
        self.center_mean_reproj_error_px = center_mean_reproj_error_px

        self.calibration_fallback_used = calibration_fallback_used
        self.rejected_calib_rmse_px = rejected_calib_rmse_px
        self.applied_calib_rmse_px = applied_calib_rmse_px

        # ── 실험 조건 스냅샷 / 시계 앵커 / 선택 기능 활성 상태 ──
        # 값 자체는 main.py가 계산해서 넘겨준다(이 클래스는 기존과 같이
        # "받은 값을 저장·출력"만 하고, config_snapshot/clock을 직접 참조하지
        # 않는다 — px_per_cm 등 기존 필드와 동일한 패턴 유지).
        self.keyboard_layout = keyboard_layout
        self.config_hash = config_hash
        self.config_json = config_json
        self.t0_utc = t0_utc
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.monitor_diagonal_inch = monitor_diagonal_inch
        self.ridge_enabled = ridge_enabled
        self.backbone_enabled = backbone_enabled

        self.input_duration_sec = None
        self.cursor_travel_distance_px = None
        self.average_cursor_speed_px_sec = None

        if use_sqpnp_corrected:
            self.correction_mode = "sqpnp_corrected"
        elif use_pose_corrected:
            self.correction_mode = "pose_corrected"
        else:
            self.correction_mode = "raw"

        self.start_timestamp = datetime.now(timezone.utc).isoformat()
        self.end_timestamp = None

        self.target_rows = []
        self._current = None

    # ── 측정 상태 조회 (main.py가 내부 변수 직접 참조하지 않도록) ──

    def is_measuring(self):
        return self._current is not None

    # ── 타깃 시작 ──────────────────────────────────────────

    def start_target(self, target_index, target_x_px, target_y_px):
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

            # ── STB-01~04용 프레임 단위 통계 ──
            "total_frames": 0,          # 전체 프레임 (모든 STB 분모)
            "face_detected_frames": 0,  # 얼굴 검출 성공 (STB-02 성공률 / STB-03 실패율)
            "gaze_valid_frames": 0,     # 시선까지 유효 (STB-04 Dropout)
            "frame_times": [],          # 각 프레임 시각 (STB-01 FPS)
        }

    # ── 매 프레임 샘플 ─────────────────────────────────────

    def add_frame(self, face_detected, gaze_valid, timestamp):
        if self._current is None:
            return

        self._current["total_frames"] += 1
        self._current["frame_times"].append(timestamp)

        if face_detected:
            self._current["face_detected_frames"] += 1

        if gaze_valid:
            self._current["gaze_valid_frames"] += 1

    def add_sample(self, gaze_x, gaze_y, iris_x, iris_y):
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
        if self._current is None:
            return

        c = self._current
        total = c["total_count"]
        valid = c["valid_count"]

        # ── STB-01~04: 프레임 단위 통계 (add_frame 기반) ──
        f_total = c["total_frames"]
        f_face = c["face_detected_frames"]
        f_gaze = c["gaze_valid_frames"]

        stb01_fps = self._compute_fps(c["frame_times"])
        stb02_landmark_rate = round(f_face / f_total, 4) if f_total > 0 else None
        stb03_face_fail = round((f_total - f_face) / f_total, 4) if f_total > 0 else None
        stb04_dropout = round((f_total - f_gaze) / f_total, 4) if f_total > 0 else None

        # STB-04 Dropout Rate
        dropout_rate = (total - valid) / total if total > 0 else None

        # 유효 프레임이 하나도 없으면 오차 계산 불가 → 실패 행으로 기록
        if valid == 0:
            self.target_rows.append({
                "run_id": self.run_id,
                "test_id": self.test_id,
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
                "stb01_fps": stb01_fps,
                "stb02_landmark_rate": stb02_landmark_rate,
                "stb03_face_fail_rate": stb03_face_fail,
                "stb04_dropout_rate": stb04_dropout,
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
            "run_id": self.run_id,
            "test_id": self.test_id,
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
            "stb01_fps": stb01_fps,
            "stb02_landmark_rate": stb02_landmark_rate,
            "stb03_face_fail_rate": stb03_face_fail,
            "stb04_dropout_rate": stb04_dropout,
            "sample_count": valid,
        })
        self._current = None

    # ── 세션 종료 / 내보내기 ───────────────────────────────

    def set_input_metrics(
        self,
        input_duration_sec,
        cursor_travel_distance_px,
        average_cursor_speed_px_sec
    ):
        self.input_duration_sec = input_duration_sec
        self.cursor_travel_distance_px = cursor_travel_distance_px
        self.average_cursor_speed_px_sec = average_cursor_speed_px_sec

    def end_session(self):
        self.end_timestamp = datetime.now(timezone.utc).isoformat()

    # sessions 행은 세 경로에서 온다 — 문장 입력 테스트를 끝까지 완료했을 때
    # (main.py의 tester.check_complete 분기), 9점 테스트는 시작했지만 완료 없이
    # 프로그램이 종료돼 안전망이 대신 저장했을 때(app_exit), 9점 테스트 자체를
    # 시작하지 않아 collector가 아예 없던 실행이 종료돼 세션 레벨 메타데이터만
    # 저장했을 때(no_test). 이 셋을 구분하지 않으면 후자들의 input_duration_sec/
    # test_id 등 결측 행이 완주 행과 섞여 입력 지표 집계를 왜곡한다.
    EXPORT_REASON_SENTENCE_COMPLETED = "sentence_completed"
    EXPORT_REASON_APP_EXIT = "app_exit"
    EXPORT_REASON_NO_TEST = "no_test"
    _VALID_EXPORT_REASONS = (
        EXPORT_REASON_SENTENCE_COMPLETED,
        EXPORT_REASON_APP_EXIT,
        EXPORT_REASON_NO_TEST,
    )

    def export_csv(
        self,
        sessions_path=None,
        accuracy_path=None,
        export_session=True,
        export_accuracy=True,
        export_reason=None,
    ):
        if sessions_path is None:
            sessions_path = f"sessions_v{self.SCHEMA_VERSION}.csv"

        if accuracy_path is None:
            accuracy_path = f"gaze_accuracy_v{self.SCHEMA_VERSION}.csv"

        if export_session:

            if export_reason not in self._VALID_EXPORT_REASONS:
                raise ValueError(
                    "export_session=True면 export_reason은 "
                    f"{self._VALID_EXPORT_REASONS} 중 하나여야 합니다: "
                    f"{export_reason!r}"
                )

            if self.end_timestamp is None:
                self.end_session()

            session_fields = [
                "run_id",
                "test_id",
                "user_id",
                "dev_version",
                "start_timestamp",
                "end_timestamp",
                "session_duration_total_ms",
                "px_per_cm",
                "calib_id",
                "calib_reproj_rmse_px",
                "correction_mode",
                "gaze_avg_window",
                "smoothing_mode",
                "input_duration_sec",
                "cursor_travel_distance_px",
                "average_cursor_speed_px_sec",
                "edge_mean_reproj_error_px",
                "center_mean_reproj_error_px",
                "calibration_fallback_used",
                "rejected_calib_rmse_px",
                "applied_calib_rmse_px",
                "keyboard_layout",
                "config_hash",
                "config_json",
                "t0_utc",
                "screen_w",
                "screen_h",
                "monitor_diagonal_inch",
                "ridge_enabled",
                "backbone_enabled",
                "export_reason",
                "schema_version",
            ]

            session_row = {
                "run_id": self.run_id,
                "test_id": self.test_id,
                "user_id": self.user_id,
                "dev_version": self.dev_version,
                "start_timestamp": self.start_timestamp,
                "end_timestamp": self.end_timestamp,
                "session_duration_total_ms": self._compute_duration_ms(),
                "px_per_cm": (
                    round(self.px_per_cm, 3)
                    if self.px_per_cm
                    else None
                ),
                "calib_id": self.calib_id,
                "calib_reproj_rmse_px": (
                    round(self.calib_reproj_rmse_px, 2)
                    if self.calib_reproj_rmse_px is not None
                    else None
                ),
                "correction_mode": self.correction_mode,
                "gaze_avg_window": self.gaze_avg_window,
                "smoothing_mode": self.smoothing_mode,
                "input_duration_sec": (
                    round(self.input_duration_sec, 2)
                    if self.input_duration_sec is not None
                    else None
                ),
                "cursor_travel_distance_px": (
                    round(self.cursor_travel_distance_px, 2)
                    if self.cursor_travel_distance_px is not None
                    else None
                ),
                "average_cursor_speed_px_sec": (
                    round(self.average_cursor_speed_px_sec, 2)
                    if self.average_cursor_speed_px_sec is not None
                    else None
                ),
                "edge_mean_reproj_error_px": (
                    round(self.edge_mean_reproj_error_px, 2)
                    if self.edge_mean_reproj_error_px is not None
                    else None
                ),
                "center_mean_reproj_error_px": (
                    round(self.center_mean_reproj_error_px, 2)
                    if self.center_mean_reproj_error_px is not None
                    else None
                ),
                "calibration_fallback_used": (
                    self.calibration_fallback_used
                ),
                "rejected_calib_rmse_px": (
                    round(self.rejected_calib_rmse_px, 2)
                    if self.rejected_calib_rmse_px is not None
                    else None
                ),
                "applied_calib_rmse_px": (
                    round(self.applied_calib_rmse_px, 2)
                    if self.applied_calib_rmse_px is not None
                    else None
                ),
                "keyboard_layout": self.keyboard_layout,
                "config_hash": self.config_hash,
                "config_json": self.config_json,
                "t0_utc": self.t0_utc,
                "screen_w": self.screen_w,
                "screen_h": self.screen_h,
                "monitor_diagonal_inch": self.monitor_diagonal_inch,
                "ridge_enabled": self.ridge_enabled,
                "backbone_enabled": self.backbone_enabled,
                "export_reason": export_reason,
                "schema_version": self.SCHEMA_VERSION,
            }

            self._append_rows(
                sessions_path,
                session_fields,
                [session_row]
            )

        if export_accuracy:

            accuracy_fields = [
                "run_id",
                "test_id",
                "target_index",
                "target_x_px",
                "target_y_px",
                "pred_x_px",
                "pred_y_px",
                "euclidean_error_px",
                "euclidean_error_cm",
                "gaze_std_x_px",
                "gaze_std_y_px",
                "iris_std_x_px",
                "iris_std_y_px",
                "dropout_rate",
                "stb01_fps",
                "stb02_landmark_rate",
                "stb03_face_fail_rate",
                "stb04_dropout_rate",
                "sample_count",
            ]

            self._append_rows(
                accuracy_path,
                accuracy_fields,
                self.target_rows
            )
    # ── 내부 헬퍼 ──────────────────────────────────────────

    def _compute_duration_ms(self):
        start = datetime.fromisoformat(self.start_timestamp)
        end = datetime.fromisoformat(self.end_timestamp)
        return int((end - start).total_seconds() * 1000)
    
    def _to_cm(self, error_px):
        if error_px is None or self.px_per_cm is None or self.px_per_cm == 0:
            return None
        return round(error_px / self.px_per_cm, 3)
    
    def _compute_fps(self, frame_times):
        if len(frame_times) < 2:
            return None

        intervals = [
            frame_times[i] - frame_times[i - 1]
            for i in range(1, len(frame_times))
        ]
        # 0 이하 간격(중복 시각 등)은 제외해 0으로 나누기 방어
        fps_values = [1.0 / dt for dt in intervals if dt > 0]

        if not fps_values:
            return None

        return round(statistics.mean(fps_values), 2)

    def _append_rows(self, path, fieldnames, rows):
        append_rows(path, fieldnames, rows)
