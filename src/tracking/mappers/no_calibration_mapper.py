"""
NoCalibrationMapper: 16점 캘리브레이션 없이 바로 ready=True가 되는 매퍼.

실제 좌표 변환은 이 클래스가 하지 않는다 — 주입받은 strategy
(src.tracking.mappers.strategies.base.NoCalibrationStrategy 구현체)에
전부 위임한다. strategy 없이는 생성 자체를 막는다(조용히 무효 좌표만
반환하며 "동작하는 것처럼" 보이는 상태를 만들지 않기 위함).
"""

from src.tracking.gaze_mapper import GazeMapper, MappingResult


class NoCalibrationMapper(GazeMapper):

    def __init__(self, strategy):
        if strategy is None:
            raise ValueError(
                "NoCalibrationMapper는 strategy 없이 생성할 수 없습니다. "
                "src.tracking.mappers.strategies.create_strategy(name)로 "
                "만든 strategy 인스턴스를 전달하세요."
            )

        self._strategy = strategy

    @property
    def ready(self):
        # 캘리브레이션 화면 자체가 없으므로 항상 True.
        # strategy 내부의 자동 anchor 워밍업은 ready가 아니라
        # map()의 MappingResult.valid로 표현된다(3~4번 답변 참고).
        return True

    @property
    def strategy(self):
        return self._strategy

    @property
    def active_method(self):
        return type(self._strategy).__name__

    def update_initialization(self, iris_x, iris_y, confidence, head_pose=None, features=None):
        # ready가 항상 True라 main.py에서 호출되지 않지만, 인터페이스 계약을
        # 지키기 위해 안전한 기본값을 반환한다.
        return 1.0

    def map(self, iris_x, iris_y, head_pose=None, features=None, sqpnp_head_pose=None):
        x, y = self._strategy.map(iris_x, iris_y, head_pose=head_pose, features=features)

        valid = x is not None and y is not None
        name = self.active_method

        return MappingResult(
            x=x,
            y=y,
            valid=valid,
            active_method=name,
            candidates={name: (x, y) if valid else None},
            metadata={"strategy_params": self._strategy.get_params()},
        )

    def reset(self):
        self._strategy.reset()

    def get_metadata(self):
        return {
            "mode": "no_calibration",
            "strategy_name": self.active_method,
            "strategy_params": self._strategy.get_params(),
        }
