from math import dist
import time
import numpy as np

from src.config import DWELL_SEC


class DwellController:

    def __init__(self):
        self.dwell_key = None
        self.dwell_start = None
        self.cooldown_end = 0
        self.hover_lock_button = None

    def update(self, gaze_x, gaze_y, buttonList):
        """
        현재 시선 좌표와 버튼 리스트를 받아 드웰 상태 갱신.

        Returns:
            (hovered_key, dwell_ratio, clicked_key)
            clicked_key는 드웰 완료 시에만 값, 나머지는 None
        """

        now = time.time()

        dwell_ratio = 0.0
        hovered_key = None
        clicked_key = None

        if gaze_x < 0 or now <= self.cooldown_end:
            self.dwell_key = None
            self.dwell_start = None
            return hovered_key, dwell_ratio, clicked_key

        closest_button = None
        closest_dist = float("inf")

        assist_radius = 35
        if self.dwell_key is not None:
             lock_radius = 60
        else:
            lock_radius = 40

        for button in buttonList:

            bx, by = button.pos
            bw, bh = button.size

            center_x = bx + bw / 2
            center_y = by + bh / 2

            dist = (
                (gaze_x - center_x) ** 2 +
                (gaze_y - center_y) ** 2
            ) ** 0.5

            if dist < closest_dist:
                closest_dist = dist
                closest_button = button



            if self.hover_lock_button is not None:

                bx, by = self.hover_lock_button.pos
                bw, bh = self.hover_lock_button.size

                center_x = bx + bw / 2
                center_y = by + bh / 2

                dist = np.hypot(
                    gaze_x - center_x,
                    gaze_y - center_y
                )

                if dist < lock_radius:
                    hovered_key = self.hover_lock_button.text
                else:
                    if dist < lock_radius:
                         hovered_key = self.hover_lock_button.text
                    else:
                        self.hover_lock_button = None


            if hovered_key is None:

                if (
                    closest_button is not None
                    and
                    closest_dist < assist_radius
                ):

                    hovered_key = closest_button.text
                    self.hover_lock_button = closest_button

        if hovered_key:

            if hovered_key != self.dwell_key:
                self.dwell_key = hovered_key
                self.dwell_start = now

            else:
                elapsed = now - self.dwell_start
                dwell_ratio = min(1.0, elapsed / DWELL_SEC)

                if dwell_ratio >= 1.0:
                    clicked_key = self.dwell_key
                    self.cooldown_end = now + 0.4
                    self.dwell_key = None
                    self.dwell_start = None

        else:
            self.dwell_key = None
            self.dwell_start = None

        return hovered_key, dwell_ratio, clicked_key