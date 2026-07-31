"""
CalibratedMapper: 기존 16점 캘리브레이션 경로를 GazeMapper 인터페이스로 감싼다.

src.tracking.calibration.Calibrator는 composition으로만 사용하고 한 줄도
복사/수정하지 않는다. 기존 main.py에 있던 raw/pose-corrected/SQPnP-corrected/
ridge-hybrid 4-way 계산 및 우선순위 선택 로직을 여기로 그대로 옮겼다
(수식과 순서는 변경 없음 — 특히 SQPnP delta 계산은 raw_sx와 같은 프레임 값을
써야 하므로 순서를 그대로 보존했다).
"""

import numpy as np

from src.tracking.calibration import Calibrator
from src.tracking.gaze_mapper import GazeMapper, MappingResult

MAX_SQPNP_DELTA_PX = 120


def _compute_calibrated_mapping(
    calibrator,
    iris_x,
    iris_y,
    head_pose,
    sqpnp_head_pose,
    features,
    active_method,
    max_sqpnp_delta_px,
):
    """
    기존 main.py 772~841행 블록을 동작 변경 없이 그대로 옮긴 순수 함수.
    """

    # 1. 기존 방식 Raw 좌표
    raw_sx, raw_sy = calibrator.map_to_screen(iris_x, iris_y)

    # 2. face center / scale 기반 iris 입력 보정 (일반 head_pose 기준)
    corrected_iris_x, corrected_iris_y = calibrator.compensate_iris_by_head_pose(
        iris_x, iris_y, head_pose
    )

    # 3. 보정된 iris 좌표를 다시 화면 좌표로 변환
    corrected_sx, corrected_sy = calibrator.map_to_screen(
        corrected_iris_x, corrected_iris_y
    )

    # SQPnP head pose 기준 보정
    sqpnp_corrected_iris_x, sqpnp_corrected_iris_y = calibrator.compensate_iris_by_head_pose(
        iris_x, iris_y, sqpnp_head_pose
    )

    sqpnp_corrected_sx, sqpnp_corrected_sy = calibrator.map_to_screen(
        sqpnp_corrected_iris_x, sqpnp_corrected_iris_y
    )

    sqpnp_delta_x = None
    sqpnp_delta_y = None

    if (
        active_method == "sqpnp_corrected"
        and raw_sx is not None
        and raw_sy is not None
        and sqpnp_corrected_sx is not None
        and sqpnp_corrected_sy is not None
    ):
        sqpnp_delta_x = np.clip(
            sqpnp_corrected_sx - raw_sx,
            -max_sqpnp_delta_px,
            max_sqpnp_delta_px,
        )
        sqpnp_delta_y = np.clip(
            sqpnp_corrected_sy - raw_sy,
            -max_sqpnp_delta_px,
            max_sqpnp_delta_px,
        )
        sqpnp_corrected_sx = int(raw_sx + sqpnp_delta_x)
        sqpnp_corrected_sy = int(raw_sy + sqpnp_delta_y)

    # 3.5. 릿지 하이브리드 좌표 (특징 벡터 → 화면 좌표)
    ridge_sx, ridge_sy = calibrator.map_to_screen_features(features)

    # 모드 우선순위: 선택된 active_method를 그대로 따름
    # 릿지 모드인데 이번 프레임 릿지 좌표가 None이면(특징 결손, 릿지 미학습 등)
    # raw로 폴백해 커서를 유지한다.
    if active_method == "ridge_hybrid" and ridge_sx is not None and ridge_sy is not None:
        sx, sy = ridge_sx, ridge_sy
    elif active_method == "ridge_hybrid":
        sx, sy = raw_sx, raw_sy
    elif active_method == "sqpnp_corrected":
        sx, sy = sqpnp_corrected_sx, sqpnp_corrected_sy
    elif active_method == "pose_corrected":
        sx, sy = corrected_sx, corrected_sy
    else:
        sx, sy = raw_sx, raw_sy

    candidates = {
        "raw": (raw_sx, raw_sy) if raw_sx is not None and raw_sy is not None else None,
        "pose_corrected": (
            (corrected_sx, corrected_sy)
            if corrected_sx is not None and corrected_sy is not None
            else None
        ),
        "sqpnp_corrected": (
            (sqpnp_corrected_sx, sqpnp_corrected_sy)
            if sqpnp_corrected_sx is not None and sqpnp_corrected_sy is not None
            else None
        ),
        "ridge_hybrid": (
            (ridge_sx, ridge_sy) if ridge_sx is not None and ridge_sy is not None else None
        ),
    }

    valid = (
        sx is not None
        and sy is not None
        and np.isfinite(sx)
        and np.isfinite(sy)
    )

    metadata = {
        "corrected_iris_x": corrected_iris_x,
        "corrected_iris_y": corrected_iris_y,
        "sqpnp_corrected_iris_x": sqpnp_corrected_iris_x,
        "sqpnp_corrected_iris_y": sqpnp_corrected_iris_y,
        "sqpnp_delta_x": sqpnp_delta_x,
        "sqpnp_delta_y": sqpnp_delta_y,
        "ridge_fitted": calibrator.ridge.fitted,
    }

    return MappingResult(
        x=sx,
        y=sy,
        valid=bool(valid),
        active_method=active_method,
        candidates=candidates,
        metadata=metadata,
    )


class CalibratedMapper(GazeMapper):

    VALID_METHODS = ("raw", "pose_corrected", "sqpnp_corrected", "ridge_hybrid")

    def __init__(self, max_sqpnp_delta_px=MAX_SQPNP_DELTA_PX):
        self._calibrator = Calibrator()
        self._active_method = "raw"
        self._max_sqpnp_delta_px = max_sqpnp_delta_px

    @property
    def ready(self):
        return self._calibrator.done

    @property
    def calibrator(self):
        """
        draw_calib_screen(), run_gaze_accuracy_test() 등 Calibrator 전용 API가
        필요한 예외 지점을 위한 탈출구. 최종 좌표 선택/후보 계산 등 일반 매핑
        책임에는 쓰지 않는다(그건 map()이 전담).
        """
        return self._calibrator

    @property
    def active_method(self):
        return self._active_method

    def set_active_method(self, method):
        if method not in self.VALID_METHODS:
            raise ValueError(f"알 수 없는 매핑 방식: {method}")

        if method == "ridge_hybrid" and not self._calibrator.ridge.fitted:
            print("[ridge] 아직 학습되지 않았습니다. 캘리브레이션(r)을 먼저 완료하세요.")

        self._active_method = method

    def update_initialization(self, iris_x, iris_y, confidence, head_pose=None, features=None):
        return self._calibrator.update(
            iris_x, iris_y, confidence, head_pose=head_pose, features=features
        )

    def map(self, iris_x, iris_y, head_pose=None, features=None, sqpnp_head_pose=None):
        effective_sqpnp_pose = (
            sqpnp_head_pose if sqpnp_head_pose is not None else head_pose
        )

        return _compute_calibrated_mapping(
            self._calibrator,
            iris_x,
            iris_y,
            head_pose,
            effective_sqpnp_pose,
            features,
            self._active_method,
            self._max_sqpnp_delta_px,
        )

    def reset(self):
        self._calibrator.reset()

    def get_metadata(self):
        return {
            "mode": "calibrated",
            "calib_id": self._calibrator.calib_id,
            "reproj_rmse_px": self._calibrator.calib_reproj_rmse_px,
            "active_method": self._active_method,
            "ridge_fitted": self._calibrator.ridge.fitted,
        }
