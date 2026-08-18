import time

from src.config import DWELL_SEC
from src.keyboard import hit_test_buttons


class DwellController:

    def __init__(self):
        self.dwell_key = None
        self.dwell_start = None
        self.cooldown_end = 0
        self.target_context = None

    def reset(self):
        """
        현재 dwell 상태를 초기화합니다.
        gaze가 유효하지 않거나 hover 상태를 초기화해야 할 때 호출합니다.
        """
        self.dwell_key = None
        self.dwell_start = None

    def set_target_context(self, context):
        """선택 대상 목록이 바뀌면 진행 중인 dwell을 취소합니다."""

        if context == self.target_context:
            return False

        self.target_context = context
        self.reset()
        return True

    def update_target(self, hovered_target, now=None):
        """이미 hit-test가 끝난 키/추천 대상을 공통 dwell로 처리합니다."""

        now = time.time() if now is None else now
        dwell_ratio = 0.0
        clicked_target = None

        if now <= self.cooldown_end:
            self.reset()
            return None, dwell_ratio, clicked_target

        if hovered_target is None:
            self.reset()
            return None, dwell_ratio, clicked_target

        if hovered_target != self.dwell_key:
            self.dwell_key = hovered_target
            self.dwell_start = now
        else:
            elapsed = now - self.dwell_start
            dwell_ratio = min(1.0, elapsed / DWELL_SEC)

            if dwell_ratio >= 1.0:
                clicked_target = self.dwell_key
                self.cooldown_end = now + 0.4
                self.reset()

        return hovered_target, dwell_ratio, clicked_target

    def update(self, gaze_x, gaze_y, buttonList):
        """
        현재 시선 좌표와 버튼 리스트를 받아 드웰 상태를 갱신합니다.

        Returns:
            (hovered_key, dwell_ratio, clicked_key)

            hovered_key:
                현재 hover 중인 키

            dwell_ratio:
                dwell 진행률, 0.0 ~ 1.0

            clicked_key:
                dwell 완료 시 입력할 키
        """

        dwell_ratio = 0.0
        hovered_key = None
        clicked_key = None

        # gaze가 유효하지 않거나 cooldown 중이면 dwell 상태 초기화
        if gaze_x < 0 or gaze_y < 0:
            self.reset()
            return hovered_key, dwell_ratio, clicked_key

        # 렌더링과 같은 Button.rect를 사용한다. 키 사이 여백에서는
        # 가장 가까운 키로 보정하지 않고 어떤 키도 선택하지 않는다.
        hovered_button = hit_test_buttons(
            buttonList,
            gaze_x,
            gaze_y,
        )
        if hovered_button is not None:
            hovered_key = hovered_button.text

        return self.update_target(hovered_key)
