import time
import numpy as np

from src.config import DWELL_SEC


class DwellController:

    def __init__(self):
        self.dwell_key = None
        self.dwell_start = None
        self.cooldown_end = 0
        self.hover_lock_button = None

    def reset(self):
        """
        현재 dwell 상태를 초기화합니다.
        gaze가 유효하지 않거나 hover 상태를 초기화해야 할 때 호출합니다.
        """
        self.dwell_key = None
        self.dwell_start = None
        self.hover_lock_button = None

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

        now = time.time()

        dwell_ratio = 0.0
        hovered_key = None
        clicked_key = None

        # gaze가 유효하지 않거나 cooldown 중이면 dwell 상태 초기화
        if gaze_x < 0 or gaze_y < 0 or now <= self.cooldown_end:
            self.reset()
            return hovered_key, dwell_ratio, clicked_key

        closest_button = None
        closest_dist = float("inf")

        assist_radius = 35

        if self.dwell_key is not None:
            lock_radius = 60
        else:
            lock_radius = 40

        # 1. 가장 가까운 버튼 찾기
        for button in buttonList:

            bx, by = button.pos
            bw, bh = button.size

            center_x = bx + bw / 2
            center_y = by + bh / 2

            distance = np.hypot(
                gaze_x - center_x,
                gaze_y - center_y
            )

            if distance < closest_dist:
                closest_dist = distance
                closest_button = button

        # 2. 기존 hover lock이 있으면 유지 가능한지 확인
        if self.hover_lock_button is not None:

            bx, by = self.hover_lock_button.pos
            bw, bh = self.hover_lock_button.size

            center_x = bx + bw / 2
            center_y = by + bh / 2

            lock_dist = np.hypot(
                gaze_x - center_x,
                gaze_y - center_y
            )

            if lock_dist < lock_radius:
                hovered_key = self.hover_lock_button.text
            else:
                self.hover_lock_button = None

        # 3. hover lock이 없으면 가장 가까운 버튼을 새로 선택
        if hovered_key is None:

            if (
                closest_button is not None
                and closest_dist < assist_radius
            ):
                hovered_key = closest_button.text
                self.hover_lock_button = closest_button

        # 4. dwell 진행
        if hovered_key:

            if hovered_key != self.dwell_key:
                self.dwell_key = hovered_key
                self.dwell_start = now

            else:
                elapsed = now - self.dwell_start
                dwell_ratio = min(
                    1.0,
                    elapsed / DWELL_SEC
                )

                if dwell_ratio >= 1.0:
                    clicked_key = self.dwell_key
                    self.cooldown_end = now + 0.4
                    self.reset()

        else:
            self.reset()

        return hovered_key, dwell_ratio, clicked_key