"""src/common/(clock, ids, config_snapshot) 단위 테스트.

웹캠·GUI 상호작용 없이 통과해야 한다. config_snapshot이 src.config를 import하면서
화면 해상도 감지를 위해 tkinter 루트가 잠깐 생성됐다 파괴되는 것은 config.py 자체의
기존 동작(디스플레이가 있는 환경 전제)이며, 이 테스트가 새로 추가하는 의존성이 아니다.
"""

import time
import uuid

import pytest

from src.common import clock, ids, config_snapshot


# ── clock ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_clock():
    clock._reset()
    yield
    clock._reset()


def test_now_ms_before_init_raises():
    with pytest.raises(RuntimeError):
        clock.now_ms()


def test_t0_utc_iso_before_init_raises():
    with pytest.raises(RuntimeError):
        clock.t0_utc_iso()


def test_now_ms_increases_after_init():
    # 0.05s: 이 환경의 time.monotonic() 실효 해상도(수~십수 ms)보다 확실히
    # 크게 잡아 tick 경계를 못 넘는 flaky 실패를 방지한다.
    clock.init()
    a = clock.now_ms()
    time.sleep(0.05)
    b = clock.now_ms()
    assert b > a


def test_t0_utc_iso_is_tz_aware():
    clock.init()
    iso = clock.t0_utc_iso()
    assert iso.endswith("+00:00") or iso.endswith("Z")


def test_duplicate_init_is_ignored_not_error():
    clock.init()
    first = clock.t0_utc_iso()
    time.sleep(0.05)
    clock.init()  # 재초기화가 아니라 경고 후 무시되어야 한다
    second = clock.t0_utc_iso()
    assert first == second


# ── ids ──────────────────────────────────────────────────────────

def test_id_functions_return_distinct_values():
    values = {ids.new_run_id(), ids.new_calib_id(), ids.new_test_id()}
    assert len(values) == 3


def test_id_functions_return_valid_uuid4():
    for factory in (ids.new_run_id, ids.new_calib_id, ids.new_test_id):
        value = factory()
        parsed = uuid.UUID(value)
        assert parsed.version == 4
        assert str(parsed) == value  # 접두어 없는 순수 uuid4 문자열


# ── config_snapshot ──────────────────────────────────────────────

def test_snapshot_hash_is_deterministic_for_same_dict():
    d = {"a": 1, "b": 2}
    assert config_snapshot.snapshot_hash(d) == config_snapshot.snapshot_hash(d)


def test_snapshot_hash_ignores_key_order():
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 2, "a": 1}
    assert config_snapshot.snapshot_hash(d1) == config_snapshot.snapshot_hash(d2)


def test_snapshot_hash_changes_when_value_changes():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 1, "b": 3}
    assert config_snapshot.snapshot_hash(d1) != config_snapshot.snapshot_hash(d2)


def test_build_snapshot_contains_all_registered_keys():
    expected_keys = {
        "smooth_alpha", "dwell_sec", "gaze_avg_window",
        "fixation_radius", "fixation_frames",
        "calib_stabilize_sec", "calib_collect_sec",
        "calib_std_x", "calib_std_y",
        "monitor_diagonal_inch", "screen_w", "screen_h",
        "ridge_alpha", "ridge_degree",
        "kalman_process_noise", "kalman_measurement_noise",
        "gaze_conf_threshold", "max_step_px",
        "fixation_center_update_rate", "dead_zone_px",
        "dwell_assist_radius", "dwell_lock_radius", "dwell_cooldown_sec",
        "max_calib_rmse_px", "mouth_mar_threshold", "mouth_hold_time_sec",
        "cheonjiin_repeat_timeout_sec",
        "targeting_dwell_sec", "targeting_timeout_sec", "targeting_prepare_sec",
    }
    snapshot = config_snapshot.build_snapshot()
    assert expected_keys <= set(snapshot.keys())
    assert "keyboard_layout" not in snapshot
    assert snapshot["cheonjiin_repeat_timeout_sec"] is None


def test_build_snapshot_excludes_px_per_cm():
    # monitor_diagonal_inch/screen_w/screen_h의 파생값이라 중복 등재하지 않는다
    # (부동소수점 결과라 환경별로 끝자리가 갈리면 같은 조건인데 해시가 달라질 위험).
    snapshot = config_snapshot.build_snapshot()
    assert "px_per_cm" not in snapshot
