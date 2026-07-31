"""
GazeMapper 공통 인터페이스.

landmarks -> raw iris/head pose 계산 -> [GazeMapper] -> 화면 좌표
경계를 표현한다. 이 경계 아래(GazePipeline, DwellController, mouth, keyboard)는
캘리브레이션 여부와 무관하게 항상 동일한 코드를 공유한다.

구현체:
    src.tracking.mappers.calibrated_mapper.CalibratedMapper
    src.tracking.mappers.no_calibration_mapper.NoCalibrationMapper
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MappingResult:
    """
    한 프레임의 매핑 결과.

    x, y:
        GazePipeline에 그대로 넘길 최종 화면 좌표. 무효하면 둘 다 None.
    valid:
        x/y가 유효한지 여부(None/NaN이 아님)를 미리 계산해 둔 편의 필드.
    active_method:
        이번 프레임에 실제로 선택된 매핑 방식 이름.
        예: "raw" / "pose_corrected" / "sqpnp_corrected" / "ridge_hybrid"
            또는 no_calibration strategy 클래스 이름.
    candidates:
        {method_name: (x, y) | None, ...}
        디버그 마커(b 키) 및 향후 비교용. 후보가 하나뿐인 매퍼는 항목이 하나만 있어도 된다.
    metadata:
        이번 프레임 한정 부가 정보(로깅/디버그 텍스트용).
        세션 단위 정적 정보는 mapper.get_metadata()에 따로 둔다.
    """

    x: float | None
    y: float | None
    valid: bool
    active_method: str
    candidates: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class GazeMapper(ABC):

    @property
    @abstractmethod
    def ready(self) -> bool:
        """True면 main.py가 map()을 호출해도 됨."""

    @abstractmethod
    def update_initialization(
        self,
        iris_x,
        iris_y,
        confidence,
        head_pose=None,
        features=None,
    ):
        """
        ready가 False인 동안 매 프레임 호출된다.
        반환값은 초기화 화면 렌더링에 쓰이는 진행률(0.0~1.0) 등.
        ready가 항상 True인 구현체(NoCalibrationMapper)는 호출되지 않는다.
        """

    @abstractmethod
    def map(
        self,
        iris_x,
        iris_y,
        head_pose=None,
        features=None,
        sqpnp_head_pose=None,
    ) -> MappingResult:
        """
        MappingResult를 반환한다.

        sqpnp_head_pose:
            CalibratedMapper 전용 예외 파라미터. estimate_sqpnp_headpose()의
            결과를 그대로 받아 SQPnP 보정 후보를 계산하는 데만 쓰인다.
            다른 구현체는 무시해도 된다.
        """

    @abstractmethod
    def reset(self):
        """'r' 키 등 재초기화 트리거."""

    @abstractmethod
    def get_metadata(self) -> dict:
        """세션 단위 정적 메타데이터(로깅/디버그용)."""
