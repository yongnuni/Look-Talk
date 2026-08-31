import csv

from src.app.performance_flow import (
    BLINK_CALIBRATION,
    DASHBOARD,
    GAZE_INPUT_TEST,
    MOUTH_CALIBRATION,
    PerformanceTestFlow,
    build_gaze_calibration_result,
)


class _CalibratorResultFixture:
    done = True
    calib_id = "calib-test"
    calib_reproj_rmse_px = 12.5
    edge_mean_reproj_error_px = 14.0
    center_mean_reproj_error_px = 8.0
    calibration_fallback_used = False
    rejected_calib_rmse_px = None
    applied_calib_rmse_px = 12.5
    last_good_calib_id = "calib-test"
    point_records = [
        {
            "iris_std_x_px": 0.01,
            "iris_std_y_px": 0.02,
            "mean_confidence": 0.9,
            "sample_count": 20,
            "used_sample_count": 18,
            "calibration_retry_count": 1,
        },
        {
            "iris_std_x_px": 0.03,
            "iris_std_y_px": 0.04,
            "mean_confidence": 0.8,
            "sample_count": 22,
            "used_sample_count": 20,
            "calibration_retry_count": 0,
        },
    ]


def test_gaze_result_adapter_only_summarizes_existing_calibrator_metrics():
    result = build_gaze_calibration_result(_CalibratorResultFixture())

    assert result["calibration_completed"] is True
    assert result["calibration_point_count"] == 2
    assert result["iris_std_x_norm_mean"] == 0.02
    assert result["iris_std_y_norm_mean"] == 0.03
    assert result["sample_count"] == 42
    assert result["used_sample_count"] == 38
    assert result["calibration_retry_count"] == 1


def test_flow_reuses_frontend_targets_and_advances_only_after_correct_confirm():
    flow = PerformanceTestFlow("run-test", random_source=lambda: 0.0)

    assert flow.complete_gaze_calibration({"calibration_point_count": 16}, 1000)
    assert flow.stage == GAZE_INPUT_TEST
    assert flow.current_target_character == "물"
    assert flow.is_entry_unlocked(1349) is False
    assert flow.is_entry_unlocked(1350) is True

    wrong = flow.confirm_input("밥", 1500)
    assert wrong.success is False
    assert flow.stage == GAZE_INPUT_TEST

    correct = flow.confirm_input("물", 2000)
    assert correct.success is True
    assert flow.stage == BLINK_CALIBRATION
    assert flow.results["gaze"].confirmation_attempts == 2
    assert flow.results["gaze"].incorrect_attempts == 1
    assert flow.results["gaze"].success_rate_percent == 50.0

    assert flow.complete_blink_calibration({"open_ear_median": 0.3}, 3000)
    assert flow.current_target_character == "밥"
    flow.confirm_input("밥", 5000)
    assert flow.stage == MOUTH_CALIBRATION

    assert flow.complete_mouth_calibration({"mar_baseline": 0.2}, 6000)
    assert flow.current_target_character == "집"
    flow.confirm_input("집", 7000)

    assert flow.stage == DASHBOARD
    assert flow.get_recommended_input_mode() == "mouth"
    assert set(flow.targets.values()) == {"물", "밥", "집"}


def test_results_export_required_attempt_fields_and_summary(tmp_path):
    flow = PerformanceTestFlow("run-export", random_source=lambda: 0.0)
    flow.complete_gaze_calibration({"calibration_point_count": 16}, 0)
    flow.confirm_input("물", 1000)
    flow.complete_blink_calibration({"close_threshold": 0.17}, 2000)
    flow.confirm_input("밥", 3000)
    flow.complete_mouth_calibration({"open_threshold": 0.4}, 4000)
    flow.confirm_input("집", 5000)

    assert flow.export_results(
        keyboard_layout="qwerty",
        user_id="tester",
        t0_utc="2026-01-01T00:00:00Z",
        runtime_quality={"stb01_fps": 30.0},
        output_dir=str(tmp_path),
    )
    assert flow.export_results(
        "qwerty", "tester", "2026-01-01T00:00:00Z", output_dir=str(tmp_path)
    ) is False

    with open(
        tmp_path / "input_method_test_results_v1.0.csv",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert {
        "target_character",
        "selected_character",
        "success",
        "input_started_at_ms",
        "selection_completed_at_ms",
        "input_duration_ms",
        "input_mode",
    } <= set(rows[0])

    with open(
        tmp_path / "performance_flow_summary_v1.0.csv",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        summary = next(csv.DictReader(handle))

    assert summary["recommended_input_mode"] == "gaze"
    assert summary["stb01_fps"] == "30.0"
