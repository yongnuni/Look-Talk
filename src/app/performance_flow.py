"""Calibration/input-performance orchestration for the existing main loop."""

from dataclasses import asdict, dataclass
import os
import random
import statistics
from typing import Optional

from src.common import clock
from src.metrics.csv_export import append_rows


INPUT_TEST_CHARACTERS = ("물", "밥", "집")
INPUT_TEST_ENTRY_LOCK_MS = 350.0
INPUT_TEST_SKIP_KEY = "s"

INPUT_TEST_STATUS_COMPLETED = "completed"
INPUT_TEST_STATUS_SKIPPED = "skipped"

GAZE_CALIBRATION = "gaze_calibration"
GAZE_INPUT_TEST = "gaze_input_test"
BLINK_CALIBRATION = "blink_calibration"
BLINK_INPUT_TEST = "blink_input_test"
MOUTH_CALIBRATION = "mouth_calibration"
MOUTH_INPUT_TEST = "mouth_input_test"
DASHBOARD = "dashboard"

INPUT_TEST_STAGES = {
    GAZE_INPUT_TEST: "gaze",
    BLINK_INPUT_TEST: "blink",
    MOUTH_INPUT_TEST: "mouth",
}


def build_gaze_calibration_result(calibrator):
    """기존 ``Calibrator``가 이미 계산·저장한 값만 결과로 노출한다."""
    point_records = list(getattr(calibrator, "point_records", ()))

    def _mean_of(field):
        values = [
            record.get(field)
            for record in point_records
            if record.get(field) is not None
        ]
        return round(statistics.mean(values), 5) if values else None

    calibration_fallback_used = bool(
        getattr(calibrator, "calibration_fallback_used", False)
    )
    attempted_calib_id = getattr(calibrator, "calib_id", None)
    last_good_calib_id = getattr(calibrator, "last_good_calib_id", None)
    applied_calib_id = (
        last_good_calib_id
        if calibration_fallback_used and last_good_calib_id is not None
        else attempted_calib_id
    )

    return {
        "calibration_completed": bool(getattr(calibrator, "done", False)),
        "calibration_point_count": len(point_records),
        "calib_id": applied_calib_id,
        "attempted_calib_id": attempted_calib_id,
        "calib_reproj_rmse_px": getattr(
            calibrator, "calib_reproj_rmse_px", None
        ),
        "edge_mean_reproj_error_px": getattr(
            calibrator, "edge_mean_reproj_error_px", None
        ),
        "center_mean_reproj_error_px": getattr(
            calibrator, "center_mean_reproj_error_px", None
        ),
        "iris_std_x_norm_mean": _mean_of("iris_std_x_px"),
        "iris_std_y_norm_mean": _mean_of("iris_std_y_px"),
        "mean_confidence": _mean_of("mean_confidence"),
        "sample_count": sum(
            record.get("sample_count", 0) for record in point_records
        ),
        "used_sample_count": sum(
            record.get("used_sample_count", record.get("sample_count", 0))
            for record in point_records
        ),
        "calibration_retry_count": sum(
            record.get("calibration_retry_count", 0)
            for record in point_records
        ),
        "calibration_fallback_used": calibration_fallback_used,
        "rejected_calib_rmse_px": getattr(
            calibrator, "rejected_calib_rmse_px", None
        ),
        "applied_calib_rmse_px": getattr(
            calibrator, "applied_calib_rmse_px", None
        ),
    }


@dataclass(frozen=True)
class InputTestAttempt:
    run_id: str
    attempt_index: int
    target_character: str
    selected_character: str
    success: bool
    input_started_at_ms: float
    selection_completed_at_ms: float
    input_duration_ms: float
    input_mode: str


@dataclass(frozen=True)
class InputMethodTestResult:
    input_mode: str
    status: str
    target_character: str
    selected_character: Optional[str]
    input_started_at_ms: Optional[float]
    selection_completed_at_ms: Optional[float]
    input_duration_ms: Optional[float]
    confirmation_attempts: Optional[int]
    incorrect_attempts: Optional[int]
    success_rate_percent: Optional[float]


class PerformanceTestFlow:
    """Advance existing calibrators and selectors without owning their logic."""

    SCHEMA_VERSION = "1.1"

    def __init__(self, run_id, random_source=None):
        self.run_id = run_id
        self.random_source = random_source or random.random
        self.stage = GAZE_CALIBRATION
        self.gaze_result = None
        self.blink_result = None
        self.mouth_result = None
        self.targets = {"gaze": None, "blink": None, "mouth": None}
        self.results = {"gaze": None, "blink": None, "mouth": None}
        self.attempts = []
        self._test_started_at_ms = None
        self._exported = False

    @property
    def current_input_mode(self):
        return INPUT_TEST_STAGES.get(self.stage)

    @property
    def current_target_character(self):
        mode = self.current_input_mode
        return self.targets[mode] if mode is not None else None

    @property
    def is_input_test(self):
        return self.current_input_mode is not None

    def complete_gaze_calibration(self, result, now_ms=None):
        if self.stage != GAZE_CALIBRATION:
            return False
        self.gaze_result = dict(result)
        self._start_input_test("gaze", GAZE_INPUT_TEST, now_ms)
        return True

    def complete_blink_calibration(self, result, now_ms=None):
        if self.stage != BLINK_CALIBRATION:
            return False
        self.blink_result = dict(result)
        self._start_input_test("blink", BLINK_INPUT_TEST, now_ms)
        return True

    def complete_mouth_calibration(self, result, now_ms=None):
        if self.stage != MOUTH_CALIBRATION:
            return False
        self.mouth_result = dict(result)
        self._start_input_test("mouth", MOUTH_INPUT_TEST, now_ms)
        return True

    def _start_input_test(self, mode, stage, now_ms):
        self.targets[mode] = self._pick_target_character()
        self.stage = stage
        self._test_started_at_ms = self._now_ms(now_ms)

    def _pick_target_character(self):
        used = {value for value in self.targets.values() if value is not None}
        available = [char for char in INPUT_TEST_CHARACTERS if char not in used]
        if not available:
            raise RuntimeError("입력 테스트에 사용할 목표 글자가 남아 있지 않습니다.")

        sample = self.random_source()
        if not isinstance(sample, (int, float)) or sample != sample:
            sample = 0.0
        sample = min(max(float(sample), 0.0), 1.0 - 1e-12)
        return available[int(sample * len(available))]

    def is_entry_unlocked(self, now_ms=None):
        if not self.is_input_test or self._test_started_at_ms is None:
            return False
        return (
            self._now_ms(now_ms) - self._test_started_at_ms
            >= INPUT_TEST_ENTRY_LOCK_MS
        )

    def confirm_input(self, selected_character, now_ms=None):
        mode = self.current_input_mode
        target = self.current_target_character
        if mode is None or target is None or self._test_started_at_ms is None:
            return None

        completed_at_ms = self._now_ms(now_ms)
        success = selected_character == target
        mode_attempts = [a for a in self.attempts if a.input_mode == mode]
        attempt = InputTestAttempt(
            run_id=self.run_id,
            attempt_index=len(mode_attempts) + 1,
            target_character=target,
            selected_character=selected_character,
            success=success,
            input_started_at_ms=self._test_started_at_ms,
            selection_completed_at_ms=completed_at_ms,
            input_duration_ms=max(0.0, completed_at_ms - self._test_started_at_ms),
            input_mode=mode,
        )
        self.attempts.append(attempt)

        if not success:
            return attempt

        attempts = [a for a in self.attempts if a.input_mode == mode]
        result = InputMethodTestResult(
            input_mode=mode,
            status=INPUT_TEST_STATUS_COMPLETED,
            target_character=target,
            selected_character=selected_character,
            input_started_at_ms=self._test_started_at_ms,
            selection_completed_at_ms=completed_at_ms,
            input_duration_ms=attempt.input_duration_ms,
            confirmation_attempts=len(attempts),
            incorrect_attempts=sum(not item.success for item in attempts),
            success_rate_percent=100.0 / len(attempts),
        )
        self.results[mode] = result
        self._test_started_at_ms = None
        self._advance_from_input_test(mode)

        return attempt

    def skip_current_input_test(self):
        mode = self.current_input_mode
        target = self.current_target_character
        if mode is None or target is None or self._test_started_at_ms is None:
            return False

        self.results[mode] = InputMethodTestResult(
            input_mode=mode,
            status=INPUT_TEST_STATUS_SKIPPED,
            target_character=target,
            selected_character=None,
            input_started_at_ms=None,
            selection_completed_at_ms=None,
            input_duration_ms=None,
            confirmation_attempts=None,
            incorrect_attempts=None,
            success_rate_percent=None,
        )
        self._test_started_at_ms = None
        self._advance_from_input_test(mode)
        return True

    def _advance_from_input_test(self, mode):
        if mode == "gaze":
            self.stage = BLINK_CALIBRATION
        elif mode == "blink":
            self.stage = MOUTH_CALIBRATION
        else:
            self.stage = DASHBOARD

    def get_recommended_input_mode(self):
        completed = [
            result
            for result in self.results.values()
            if result is not None
            and result.status == INPUT_TEST_STATUS_COMPLETED
        ]
        if not completed:
            return None

        order = {"gaze": 0, "blink": 1, "mouth": 2}
        return min(
            completed,
            key=lambda result: (
                -result.success_rate_percent,
                result.input_duration_ms,
                order[result.input_mode],
            ),
        ).input_mode

    def get_dashboard_summary(self, runtime_quality=None):
        runtime_quality = runtime_quality or {}
        calibration_fallback_count = sum(
            (
                bool((self.gaze_result or {}).get("calibration_fallback_used")),
                bool((self.blink_result or {}).get("calibration_failed_fallback")),
            )
        )
        return {
            "calibrations": {
                "gaze": self.gaze_result or {},
                "blink": self.blink_result or {},
                "mouth": self.mouth_result or {},
                "completed_count": sum(
                    value is not None
                    for value in (self.gaze_result, self.blink_result, self.mouth_result)
                ),
                "fallback_count": calibration_fallback_count,
            },
            "tests": {
                mode: asdict(result) if result is not None else None
                for mode, result in self.results.items()
            },
            "runtime_quality": dict(runtime_quality),
            "recommended_input_mode": self.get_recommended_input_mode(),
        }

    def export_results(
        self,
        keyboard_layout,
        user_id,
        t0_utc,
        runtime_quality=None,
        output_dir="gaze_accuracy_results",
    ):
        if self._exported or self.stage != DASHBOARD:
            return False

        os.makedirs(output_dir, exist_ok=True)
        attempt_fields = [
            "run_id",
            "attempt_index",
            "target_character",
            "selected_character",
            "success",
            "input_started_at_ms",
            "selection_completed_at_ms",
            "input_duration_ms",
            "input_mode",
            "keyboard_layout",
            "schema_version",
        ]
        attempt_rows = []
        for attempt in self.attempts:
            row = asdict(attempt)
            row["input_started_at_ms"] = round(row["input_started_at_ms"], 1)
            row["selection_completed_at_ms"] = round(
                row["selection_completed_at_ms"], 1
            )
            row["input_duration_ms"] = round(row["input_duration_ms"], 1)
            row["keyboard_layout"] = keyboard_layout
            row["schema_version"] = self.SCHEMA_VERSION
            attempt_rows.append(row)

        append_rows(
            os.path.join(
                output_dir,
                f"input_method_test_results_v{self.SCHEMA_VERSION}.csv",
            ),
            attempt_fields,
            attempt_rows,
        )

        runtime_quality = runtime_quality or {}
        gaze = self.gaze_result or {}
        blink = self.blink_result or {}
        mouth = self.mouth_result or {}
        summary_row = {
            "run_id": self.run_id,
            "user_id": user_id,
            "keyboard_layout": keyboard_layout,
            "t0_utc": t0_utc,
            "gaze_calibration_completed": bool(self.gaze_result),
            "calibration_point_count": gaze.get("calibration_point_count"),
            "calib_id": gaze.get("calib_id"),
            "calib_reproj_rmse_px": gaze.get("calib_reproj_rmse_px"),
            "edge_mean_reproj_error_px": gaze.get("edge_mean_reproj_error_px"),
            "center_mean_reproj_error_px": gaze.get("center_mean_reproj_error_px"),
            "iris_std_x_norm_mean": gaze.get("iris_std_x_norm_mean"),
            "iris_std_y_norm_mean": gaze.get("iris_std_y_norm_mean"),
            "gaze_mean_confidence": gaze.get("mean_confidence"),
            "gaze_sample_count": gaze.get("sample_count"),
            "gaze_used_sample_count": gaze.get("used_sample_count"),
            "gaze_calibration_retry_count": gaze.get(
                "calibration_retry_count"
            ),
            "gaze_calibration_fallback_used": gaze.get(
                "calibration_fallback_used"
            ),
            "blink_open_ear_median": blink.get("open_ear_median"),
            "blink_closed_ear_median": blink.get("closed_ear_median"),
            "blink_close_threshold": blink.get("close_threshold"),
            "blink_open_threshold": blink.get("open_threshold"),
            "blink_total_trials": blink.get("total_trials"),
            "blink_closed_sample_count": blink.get("closed_sample_count"),
            "blink_calibration_fallback": blink.get(
                "calibration_failed_fallback"
            ),
            "mouth_mar_baseline": mouth.get("mar_baseline"),
            "mouth_open_mar": mouth.get("open_mar"),
            "mouth_open_threshold": mouth.get("open_threshold"),
            "mouth_close_threshold": mouth.get("close_threshold"),
            "mouth_success_rate": mouth.get("mouth_success_rate"),
            "mouth_consistency": mouth.get("mouth_consistency"),
            "mouth_init_score": mouth.get("mouth_init_score"),
            "mouth_activation_amplitude_mean": mouth.get(
                "activation_amplitude_mean"
            ),
            "mouth_activation_duration_mean": mouth.get(
                "activation_duration_mean"
            ),
            "mouth_open_close_speed_mean": mouth.get(
                "open_close_speed_mean"
            ),
            "mouth_contrast_ratio": mouth.get("mouth_contrast_ratio"),
            "mouth_false_trigger_rate": mouth.get(
                "mouth_false_trigger_rate"
            ),
            "mouth_amplitude_decay": mouth.get("amplitude_decay"),
            "mouth_min_hold_duration": mouth.get(
                "mouth_min_hold_duration"
            ),
            "mouth_total_trials": mouth.get("total_trials"),
            "mouth_success_count": mouth.get("success_count"),
            "mouth_false_trigger_count": mouth.get("false_trigger_count"),
            "stb01_fps": runtime_quality.get("stb01_fps"),
            "stb02_landmark_rate": runtime_quality.get("stb02_landmark_rate"),
            "stb03_face_fail_rate": runtime_quality.get("stb03_face_fail_rate"),
            "stb04_dropout_rate": runtime_quality.get("stb04_dropout_rate"),
            "recommended_input_mode": self.get_recommended_input_mode(),
            "schema_version": self.SCHEMA_VERSION,
        }
        for mode, result in self.results.items():
            summary_row[f"{mode}_status"] = result.status if result else None
            summary_row[f"{mode}_target_character"] = (
                result.target_character if result else None
            )
            summary_row[f"{mode}_selected_character"] = (
                result.selected_character if result else None
            )
            summary_row[f"{mode}_success_rate_percent"] = (
                round(result.success_rate_percent, 2)
                if result and result.success_rate_percent is not None
                else None
            )
            summary_row[f"{mode}_average_input_time_sec"] = (
                round(result.input_duration_ms / 1000.0, 3)
                if result and result.input_duration_ms is not None
                else None
            )
            summary_row[f"{mode}_incorrect_attempts"] = (
                result.incorrect_attempts if result else None
            )

        append_rows(
            os.path.join(
                output_dir,
                f"performance_flow_summary_v{self.SCHEMA_VERSION}.csv",
            ),
            list(summary_row.keys()),
            [summary_row],
        )
        self._exported = True
        return True

    @staticmethod
    def _now_ms(now_ms):
        return clock.now_ms() if now_ms is None else float(now_ms)
