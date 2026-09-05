"""BlinkCalibration 정책 검증.

looktalk-frontend의 src/features/calibration/inputCalibration.ts에 있는
BlinkCalibrationSession과 같은 정책을 유지한다: 눈을 뜬 상태 수집 → 깜빡임 1회를
5회 반복, median 기반 threshold, 대비가 너무 작으면 detector 기본값.

프론트엔드와 맞춘 부분:
  - 대기 창이 그냥 만료된 회차의 최소값은 closed 표본에 넣지 않는다.
  - 3회 연속 깜빡임을 놓치면 실패 상태로 멈추고 재측정/기본값을 고르게 한다.
  - 얼굴이 보이지 않는 구간은 단계 경과 시간에서 뺀다.
"""

import pytest

from src.calibrations.blink_calibration import BlinkCalibration


def _collect_open(calibration, open_ear, start):
    """open 단계를 통과시키고 blink 단계 시작 시각을 돌려준다."""

    for index in range(14):
        calibration.update(open_ear, now=start + index * 0.05)

    calibration.update(open_ear, now=start + 1.21)
    assert calibration.state == "blink"

    return calibration.state_started_at


def _run_round(calibration, open_ear, closed_ear, start):
    """open 수집 → 깜빡임 1회 → rest까지 진행하고 다음 시작 시각을 돌려준다."""

    blink_started = _collect_open(calibration, open_ear, start)

    calibration.update(closed_ear, now=blink_started + 0.1)
    calibration.update(open_ear, now=blink_started + 0.2)

    if calibration.state == "rest":
        calibration.update(
            open_ear,
            now=calibration.state_started_at + 0.61,
        )

    return calibration.state_started_at


def _run_timeout_round(calibration, open_ear, start):
    """눈을 뜬 채로 대기 창을 만료시켜 한 회차를 놓친다."""

    blink_started = _collect_open(calibration, open_ear, start)
    calibration.update(open_ear, now=blink_started + 2.01)

    return calibration.state_started_at


def test_blink_policy_calculates_personal_thresholds():
    calibration = BlinkCalibration()
    calibration.reset(now=0.0)

    start = 0.0
    for _ in range(5):
        start = _run_round(calibration, 0.3, 0.1, start)

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


def test_blink_policy_uses_existing_detector_defaults_for_small_span():
    """깜빡임은 확인됐지만 open/closed 절대 차이가 0.05 미만이면 기본값."""

    calibration = BlinkCalibration()
    calibration.reset(now=0.0)

    start = 0.0
    for _ in range(5):
        start = _run_round(calibration, 0.10, 0.055, start)

    result = calibration.get_result_dict()
    assert result["calibration_failed_fallback"] is True
    assert result["close_threshold"] == pytest.approx(0.18)
    assert result["open_threshold"] == pytest.approx(0.22)


def test_timed_out_round_is_not_counted_as_a_closed_sample():
    """대기 창이 만료된 회차의 최소값은 눈을 뜬 채 관측한 값이라 버린다.

    이걸 closed 표본에 섞으면 closed_median이 위로 끌려 올라가 threshold가
    통째로 어긋난다.
    """

    calibration = BlinkCalibration()
    calibration.reset(now=0.0)

    start = _run_timeout_round(calibration, 0.3, 0.0)
    assert calibration.failed_attempts == 1
    assert calibration.current_trial_index == 0
    assert calibration.closed_minima == []

    for _ in range(5):
        start = _run_round(calibration, 0.3, 0.1, start)

    result = calibration.get_result_dict()
    assert calibration.failed_attempts == 0
    assert result["closed_sample_count"] == 5
    assert result["closed_ear_median"] == pytest.approx(0.1)


def test_three_missed_blinks_stop_with_a_failed_state():
    calibration = BlinkCalibration()
    calibration.reset(now=0.0)

    start = 0.0
    for _ in range(3):
        start = _run_timeout_round(calibration, 0.3, start)

    assert calibration.failed is True
    assert calibration.state == "failed"
    assert calibration.done is False
    assert calibration.get_result_dict() is None


def test_continue_with_defaults_after_failure():
    calibration = BlinkCalibration()
    calibration.reset(now=0.0)

    start = 0.0
    for _ in range(3):
        start = _run_timeout_round(calibration, 0.3, start)

    calibration.continue_with_defaults()

    result = calibration.get_result_dict()
    assert calibration.done is True
    assert calibration.failed is False
    assert result["calibration_failed_fallback"] is True
    assert result["close_threshold"] == pytest.approx(0.18)
    assert result["open_threshold"] == pytest.approx(0.22)


def test_time_without_a_face_does_not_expire_the_blink_window():
    """얼굴이 사라진 10초 동안 깜빡임 대기 창(2초)이 만료되면 안 된다."""

    calibration = BlinkCalibration()
    calibration.reset(now=0.0)

    _collect_open(calibration, 0.3, 0.0)

    calibration.update(None, now=1.5)
    calibration.update(None, now=11.5)
    calibration.update(0.3, now=11.55)

    assert calibration.state == "blink"
    assert calibration.failed_attempts == 0

    # 얼굴이 돌아온 뒤 정상적으로 깜빡이면 그대로 성공 처리된다.
    calibration.update(0.1, now=11.6)
    calibration.update(0.3, now=11.7)

    assert calibration.state == "rest"
    assert calibration.closed_minima == [pytest.approx(0.1)]
