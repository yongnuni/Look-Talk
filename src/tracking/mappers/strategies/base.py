"""
NoCalibrationStrategy: no_calibration 모드의 실제 좌표 변환 알고리즘이
구현해야 하는 최소 인터페이스.

NoCalibrationMapper는 이 인터페이스만 알고, 구체적인 계산 방식은 전혀 모른다.
그래서 strategy를 교체해도 NoCalibrationMapper/main.py는 수정할 필요가 없다.
"""

from abc import ABC, abstractmethod


class NoCalibrationStrategy(ABC):

    @abstractmethod
    def map(self, iris_x, iris_y, head_pose=None, features=None):
        """
        (screen_x, screen_y) 또는 (None, None)을 반환한다.
        Calibrator.map_to_screen()과 동일한 계약(무효 시 (None, None), 유효 시
        화면 범위로 clip된 좌표)을 지킨다.
        """

    @abstractmethod
    def reset(self):
        """내부 상태(예: 자동 anchor)를 초기화한다."""

    @abstractmethod
    def get_params(self) -> dict:
        """현재 파라미터 값을 dict로 반환한다(로깅/디버그용)."""
