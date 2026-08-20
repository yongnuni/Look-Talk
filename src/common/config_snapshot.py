"""
실험 조건(파라미터) 스냅샷.

배경: 파라미터가 config.py / 각 모듈 하드코딩 / 호출부 인자 세 곳에 흩어져 있어
config.py 파일 하나만 해시해서는 실제 동작에 영향을 주는 파라미터 변경(dwell 반경,
Kalman 계수 등)을 감지하지 못한다(docs/current_state_report.md 2-5절 참고). 그래서
config.py에서 가져올 수 있는 값은 가져오고, 나머지는 코드에 박힌 리터럴을 그대로
이 파일에 옮겨 적어(등재) 유효값 전체를 dict로 모은 뒤 해시한다.

*** 하드코딩 등재값은 원본 코드와 자동 동기화되지 않는다. 해당 상수를 코드에서
변경할 때는 이 파일도 함께 수정해야 하며, 그렇지 않으면 config_hash가 실험 조건
변화를 놓친다. 근본 해결(상수의 config 이관)은 개편 완료 후 별도 과제로 남긴다. ***

keyboard_layout은 이 스냅샷에 넣지 않는다 — 1단계에서 sessions 행의 독립 컬럼으로
별도 기록하기로 확정됐다. keyboard_layout은 파라미터(설정값)라기보다 세션마다
달라지는 실행 조건이라, 스냅샷 안에 섞으면 같은 파라미터 조합인데 레이아웃만 다른
두 세션이 서로 다른 해시를 받아 "실험 조건이 바뀌었다"는 잘못된 신호를 준다
(중복 기록 방지).

px_per_cm도 등재하지 않는다 — monitor_diagonal_inch/screen_w/screen_h 세 값에서
계산되는 파생값이라 원본과 정보가 중복되고, 부동소수점 나눗셈 결과라 실행 환경에
따라 끝자리가 달라지면 같은 조건인데도 해시가 갈라질 수 있기 때문이다. 세 원본만
등재하면 충분하다.
"""

import hashlib
import json

from src.config import (
    SMOOTH_ALPHA,
    DWELL_SEC,
    GAZE_AVG_WINDOW,
    FIXATION_RADIUS,
    FIXATION_FRAMES,
    CALIB_STABILIZE_SEC,
    CALIB_COLLECT_SEC,
    CALIB_STD_X,
    CALIB_STD_Y,
    MONITOR_DIAGONAL_INCH,
    SCREEN_W,
    SCREEN_H,
    RIDGE_ALPHA,
    RIDGE_DEGREE,
)


def build_snapshot():
    """등재 목록의 {이름: 현재값} dict를 반환한다."""
    return {
        # ── config.py에서 import (값 변경이 자동 반영됨) ──
        "smooth_alpha": SMOOTH_ALPHA,
        "dwell_sec": DWELL_SEC,
        "gaze_avg_window": GAZE_AVG_WINDOW,
        "fixation_radius": FIXATION_RADIUS,
        "fixation_frames": FIXATION_FRAMES,
        "calib_stabilize_sec": CALIB_STABILIZE_SEC,
        "calib_collect_sec": CALIB_COLLECT_SEC,
        "calib_std_x": CALIB_STD_X,
        "calib_std_y": CALIB_STD_Y,
        # px_per_cm은 등재하지 않는다 — 아래 세 값에서 계산되는 파생값이라
        # 정보가 중복되고, 부동소수점 결과라 환경별로 끝자리가 갈릴 수 있다.
        "monitor_diagonal_inch": MONITOR_DIAGONAL_INCH,
        "screen_w": SCREEN_W,
        "screen_h": SCREEN_H,
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_degree": RIDGE_DEGREE,

        # ── 하드코딩 리터럴 등재 (출처 위치는 각 줄 주석 참고) ──
        "kalman_process_noise": 0.003,        # src/tracking/gaze_pipeline.py:48
        "kalman_measurement_noise": 0.3,      # src/tracking/gaze_pipeline.py:52
        "gaze_conf_threshold": 0.3,           # src/tracking/gaze_pipeline.py:129
        "max_step_px": 50.0,                  # src/tracking/gaze_pipeline.py:225
        "fixation_center_update_rate": 0.05,  # src/tracking/gaze_pipeline.py:252,255
        "dead_zone_px": [10, 15, 25],         # src/tracking/gaze_pipeline.py:207-212
        "dwell_assist_radius": 35,            # src/tracking/dwell.py:55
        "dwell_lock_radius": [40, 60],        # src/tracking/dwell.py:57-60 ([hover_lock 없음, hover_lock 있음] 순)
        "dwell_cooldown_sec": 0.4,            # src/tracking/dwell.py:125
        "max_calib_rmse_px": 150.0,           # src/tracking/calibration.py:615
        "mouth_mar_threshold": 0.30,          # src/tracking/mouth.py:64
        "mouth_hold_time_sec": 0.3,           # src/tracking/mouth.py:65
        "cheonjiin_repeat_timeout_sec": None, # 시간 제한 없는 확정 전 순환
        "targeting_dwell_sec": 1.0,           # main.py:552 (TargetingTestRunner 호출부)
        "targeting_timeout_sec": 5.0,         # main.py:553 (TargetingTestRunner 호출부)
        "targeting_prepare_sec": 2.0,         # tests/targeting_test_runner.py:42 (클래스 기본값 — main.py 호출부는 이 인자를 넘기지 않는다)
    }


def snapshot_hash(d):
    """dict를 canonical JSON으로 직렬화해 sha256 앞 12자(hex)를 반환한다.

    json.dumps(sort_keys=True)로 키 순서 차이를 제거한다. 값 자체는 가공하지
    않고 파이썬 기본 직렬화 경로만 거치므로, 같은 코드라면 항상 같은 해시가 나온다.
    """
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
