"""
GazeMapper 생성 factory. main.py는 이 모듈만 알면 되고, CalibratedMapper/
NoCalibrationMapper/strategy 구현 세부사항은 몰라도 된다.

CLI/설정에서 쓰는 모드 문자열은 이 파일의 MODE_CALIBRATED/MODE_NO_CALIBRATION
두 상수로만 정의한다(프로젝트 전체에서 "no_calibration" 한 가지 표기로 통일).
"""

from src.tracking.mappers.calibrated_mapper import CalibratedMapper
from src.tracking.mappers.no_calibration_mapper import NoCalibrationMapper
from src.tracking.mappers.strategies import create_strategy, available_strategies

MODE_CALIBRATED = "calibrated"
MODE_NO_CALIBRATION = "no_calibration"

AVAILABLE_MODES = (MODE_CALIBRATED, MODE_NO_CALIBRATION)

DEFAULT_NO_CALIBRATION_STRATEGY = "head_pose_relative_iris"


def create_mapper(mode=MODE_CALIBRATED, strategy_name=None):
    """
    mode:
        "calibrated" | "no_calibration"
    strategy_name:
        no_calibration 모드에서 사용할 strategy 이름.
        생략하면 DEFAULT_NO_CALIBRATION_STRATEGY를 사용한다.
        calibrated 모드에서는 무시된다.

    등록된 strategy가 없는데 no_calibration을 요청하면 여기서 명확한
    예외로 막는다(NoCalibrationMapper가 "일단 실행되는 것처럼" 보이는
    상태로 넘어가지 않도록 하기 위함 — main.py의 CLI 파싱 단계에서 이미
    한 번 더 걸러지지만, factory를 직접 호출하는 다른 코드도 안전해야 한다).
    """

    if mode == MODE_CALIBRATED:
        return CalibratedMapper()

    if mode == MODE_NO_CALIBRATION:
        name = strategy_name or DEFAULT_NO_CALIBRATION_STRATEGY

        if name not in available_strategies():
            available = ", ".join(available_strategies()) or "(없음)"
            raise ValueError(
                f"'{name}'은(는) 등록된 no_calibration strategy가 아닙니다. "
                f"사용 가능한 strategy: {available}"
            )

        strategy = create_strategy(name)
        return NoCalibrationMapper(strategy)

    raise ValueError(
        f"알 수 없는 gaze mapper 모드: {mode!r} "
        f"(사용 가능: {', '.join(AVAILABLE_MODES)})"
    )
