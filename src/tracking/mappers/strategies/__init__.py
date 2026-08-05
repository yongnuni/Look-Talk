"""
NoCalibration strategy registry.

새 strategy를 추가하려면:
    1. 이 폴더에 새 파일을 만들고 NoCalibrationStrategy를 구현한다.
    2. @register_strategy("이름")을 클래스에 붙인다.
    3. 이 파일 하단의 import 목록에 새 모듈을 추가한다(자기등록을 위해 필요).

factory.py/no_calibration_mapper.py는 이 레지스트리만 알고, 구체적인 strategy
구현은 전혀 모른다.
"""

STRATEGY_REGISTRY = {}


def register_strategy(name):
    def _decorator(cls):
        STRATEGY_REGISTRY[name] = cls
        return cls
    return _decorator


def create_strategy(name, **kwargs):
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(available_strategies()) or "(없음)"
        raise ValueError(
            f"등록되지 않은 strategy '{name}'. 사용 가능한 strategy: {available}"
        )
    return STRATEGY_REGISTRY[name](**kwargs)


def available_strategies():
    return sorted(STRATEGY_REGISTRY.keys())


# ── 여기서부터 구체적인 strategy를 import해 자기등록시킨다 ──
# (STRATEGY_REGISTRY/register_strategy가 이미 정의된 뒤라 순환 import 문제 없음)
from src.tracking.mappers.strategies.head_pose_relative_iris import (  # noqa: E402,F401
    HeadPoseRelativeIrisStrategy,
)
