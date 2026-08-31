import pytest

from src.calibrations.blink_calibration import BlinkCalibration


def _run_round(calibration, open_ear, closed_ear):
    open_started = calibration.state_started_at
    for index in range(14):
        calibration.update(open_ear, now=open_started + index * 0.05)
    calibration.update(open_ear, now=open_started + 1.21)
    assert calibration.state == "blink"

    blink_started = calibration.state_started_at
    calibration.update(closed_ear, now=blink_started + 0.1)
    calibration.update(open_ear, now=blink_started + 0.2)
    if calibration.state == "blink":
        calibration.update(open_ear, now=blink_started + 2.01)
    assert calibration.state == "rest"

    rest_started = calibration.state_started_at
    calibration.update(open_ear, now=rest_started + 0.61)


def test_restored_blink_policy_calculates_personal_thresholds():
    calibration = BlinkCalibration()
    calibration.reset(now=0.0)

    for _ in range(5):
        _run_round(calibration, open_ear=0.3, closed_ear=0.1)

    result = calibration.get_result_dict()
    assert calibration.done is True
    assert result == {
        "close_threshold": pytest.approx(0.17),
        "open_threshold": pytest.approx(0.21),
        "open_ear_median": pytest.approx(0.3),
        "closed_ear_median": pytest.approx(0.1),
        "calibration_failed_fallback": False,
        "total_trials": 5,
        "closed_sample_count": 5,
    }


def test_restored_blink_policy_uses_existing_detector_defaults_for_small_span():
    calibration = BlinkCalibration()
    calibration.reset(now=0.0)

    for _ in range(5):
        _run_round(calibration, open_ear=0.3, closed_ear=0.27)

    result = calibration.get_result_dict()
    assert result["calibration_failed_fallback"] is True
    assert result["close_threshold"] == pytest.approx(0.18)
    assert result["open_threshold"] == pytest.approx(0.22)
